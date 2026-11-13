use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

# Experimental RaBitQ codes stored instead of vectors in HNSW indexes
# (rabitq_bits index option). Inserts, vacuum, and on-disk builds use
# estimated distances.
# Recall thresholds are set below values measured with 50 queries on this data

my $node;
my @queries = ();
my @expected;
my $limit = 10;
my $dim = 128;

sub test_recall
{
	my ($min, $operator, $ef_search, $name) = @_;
	my $correct = 0;
	my $total = 0;

	for my $i (0 .. $#queries)
	{
		my $actual = $node->safe_psql("postgres", qq(
			SET enable_seqscan = off;
			SET hnsw.ef_search = $ef_search;
			SELECT i FROM tst ORDER BY v $operator '$queries[$i]' LIMIT $limit;
		));
		my @actual_ids = split("\n", $actual);
		my %actual_set = map { $_ => 1 } @actual_ids;

		my @expected_ids = split("\n", $expected[$i]);

		foreach (@expected_ids)
		{
			if (exists($actual_set{$_}))
			{
				$correct++;
			}
			$total++;
		}
	}

	my $recall = $correct / $total;
	note("$name $operator ef_search=$ef_search recall=$recall");
	cmp_ok($recall, ">=", $min, "$name $operator ef_search=$ef_search");
}

sub test_order
{
	my ($operator) = @_;

	# Results must come back in exact distance order
	my $res = $node->safe_psql("postgres", qq(
		SET enable_seqscan = off;
		SELECT bool_and(d >= prev) FROM (
			SELECT d, lag(d, 1, '-Infinity') OVER (ORDER BY n) AS prev FROM (
				SELECT v $operator '$queries[0]' AS d, row_number() OVER () AS n
				FROM (SELECT v FROM tst ORDER BY v $operator '$queries[0]' LIMIT 40) s
			) t
		) u;
	));
	is($res, "t", "$operator ordered");
}

sub get_expected
{
	my ($operator) = @_;

	@expected = ();
	foreach (@queries)
	{
		my $res = $node->safe_psql("postgres", qq(
			SET enable_indexscan = off;
			SELECT i FROM tst ORDER BY v $operator '$_' LIMIT $limit;
		));
		push(@expected, $res);
	}
}

sub random_array_sql
{
	my ($n) = @_;
	return "ARRAY(SELECT random() - 0.5 + i * 0 FROM generate_series(1, $n))";
}

# Initialize node
$node = PostgreSQL::Test::Cluster->new('node');
$node->init;
$node->start;

# Create table
$node->safe_psql("postgres", "CREATE EXTENSION vector;");
$node->safe_psql("postgres", "CREATE TABLE tst (i int4, v vector($dim));");
$node->safe_psql("postgres", "INSERT INTO tst SELECT i, " . random_array_sql($dim) . " FROM generate_series(1, 5000) i;");
$node->safe_psql("postgres", "CREATE TABLE tst2 AS SELECT * FROM tst;");

# Generate queries
for (1 .. 20)
{
	push(@queries, "[" . join(",", map { rand() - 0.5 } (1 .. $dim)) . "]");
}

my @operators = ("<->", "<#>", "<=>");
my @opclasses = ("vector_l2_ops", "vector_ip_ops", "vector_cosine_ops");

for my $type ("vector", "halfvec")
{
	if ($type eq "halfvec")
	{
		$node->safe_psql("postgres", "ALTER TABLE tst ALTER COLUMN v TYPE halfvec($dim);");
		$node->safe_psql("postgres", "ALTER TABLE tst2 ALTER COLUMN v TYPE halfvec($dim);");
	}

	for my $i (0 .. $#operators)
	{
		my $operator = $operators[$i];
		my $opclass = $opclasses[$i];
		$opclass =~ s/^vector/halfvec/ if $type eq "halfvec";

		get_expected($operator);

		# In-memory build with exact distances (measured about 0.91-0.94 and 0.98-0.99)
		$node->safe_psql("postgres", "CREATE INDEX idx ON tst USING hnsw (v $opclass) WITH (rabitq_bits = 2);");
		test_recall(0.80, $operator, 100, "$type build");
		test_recall(0.93, $operator, 200, "$type build");
		test_order($operator);

		# Vacuum and inserts with estimated distances (measured about 0.91-0.93)
		$node->safe_psql("postgres", "DELETE FROM tst WHERE i % 2 = 0;");
		$node->safe_psql("postgres", "VACUUM tst;");
		$node->safe_psql("postgres", "INSERT INTO tst SELECT * FROM tst2 WHERE i % 2 = 0;");
		test_recall(0.80, $operator, 100, "$type vacuum");
		test_order($operator);

		$node->safe_psql("postgres", "DROP INDEX idx;");
	}
}

$node->safe_psql("postgres", "ALTER TABLE tst ALTER COLUMN v TYPE vector($dim);");
$node->safe_psql("postgres", "ALTER TABLE tst2 ALTER COLUMN v TYPE vector($dim);");
get_expected("<->");

# Index is smaller without vectors
$node->safe_psql("postgres", "CREATE INDEX plain ON tst USING hnsw (v vector_l2_ops);");
$node->safe_psql("postgres", "CREATE INDEX idx ON tst USING hnsw (v vector_l2_ops) WITH (rabitq_bits = 2);");
my $plain_size = $node->safe_psql("postgres", "SELECT pg_relation_size('plain');");
my $size = $node->safe_psql("postgres", "SELECT pg_relation_size('idx');");
cmp_ok($size, "<", $plain_size / 2, "index without vectors is smaller");
$node->safe_psql("postgres", "DROP INDEX plain;");
$node->safe_psql("postgres", "DROP INDEX idx;");

# All inserts (measured about 0.92)
$node->safe_psql("postgres", "TRUNCATE tst;");
$node->safe_psql("postgres", "CREATE INDEX idx ON tst USING hnsw (v vector_l2_ops) WITH (rabitq_bits = 2);");
$node->safe_psql("postgres", "INSERT INTO tst SELECT * FROM tst2;");
test_recall(0.80, "<->", 100, "inserts");

# Duplicates
$node->safe_psql("postgres", "INSERT INTO tst SELECT -i, v FROM tst2 WHERE i <= 100;");
my $count = $node->safe_psql("postgres", qq(
	SET enable_seqscan = off;
	SET hnsw.ef_search = 100;
	SELECT COUNT(*) FROM (SELECT i FROM tst ORDER BY v <-> (SELECT v FROM tst2 WHERE i = 1) LIMIT 2) s WHERE abs(i) = 1;
));
is($count, 2, "duplicates");
$node->safe_psql("postgres", "DELETE FROM tst WHERE i < 0;");
$node->safe_psql("postgres", "DROP INDEX idx;");

# On-disk and parallel builds
my ($ret, $stdout, $stderr) = $node->psql("postgres", qq(
	ALTER TABLE tst SET (parallel_workers = 2);
	SET client_min_messages = DEBUG;
	SET maintenance_work_mem = '2MB';
	CREATE INDEX idx ON tst USING hnsw (v vector_l2_ops) WITH (rabitq_bits = 2);
	ALTER TABLE tst RESET (parallel_workers);
));
is($ret, 0, $stderr);
like($stderr, qr/using \d+ parallel workers/);
like($stderr, qr/hnsw graph no longer fits into maintenance_work_mem/);
like($stderr, qr/built with estimated distances/);
test_recall(0.80, "<->", 100, "on-disk build");

# Crash recovery
$node->safe_psql("postgres", "INSERT INTO tst SELECT i, " . random_array_sql($dim) . " FROM generate_series(5001, 5500) i;");
$node->stop('immediate');
$node->start;
get_expected("<->");
test_recall(0.80, "<->", 100, "recovery");

# Exact distances and other bits are not available
($ret, $stdout, $stderr) = $node->psql("postgres", qq(
	SET enable_seqscan = off;
	SET hnsw.rabitq_bits = 0;
	SELECT i FROM tst ORDER BY v <-> '$queries[0]' LIMIT 1;
));
like($stderr, qr/index only supports hnsw.rabitq_bits = 2/);

($ret, $stdout, $stderr) = $node->psql("postgres", qq(
	SET enable_seqscan = off;
	SET hnsw.rabitq_bits = 4;
	SELECT i FROM tst ORDER BY v <-> '$queries[0]' LIMIT 1;
));
like($stderr, qr/index only supports hnsw.rabitq_bits = 2/);

my $res = $node->safe_psql("postgres", qq(
	SET enable_seqscan = off;
	SET hnsw.rabitq_bits = 2;
	SELECT COUNT(*) FROM (SELECT i FROM tst ORDER BY v <-> '$queries[0]' LIMIT 5) s;
));
is($res, 5);

# Iterative scans are not supported
($ret, $stdout, $stderr) = $node->psql("postgres", qq(
	SET enable_seqscan = off;
	SET hnsw.iterative_scan = relaxed_order;
	SELECT i FROM tst ORDER BY v <-> '$queries[0]' LIMIT 1;
));
like($stderr, qr/iterative scans are not supported/);
$node->safe_psql("postgres", "DROP INDEX idx;");

# More dimensions than indexes with vectors support
$node->safe_psql("postgres", "CREATE TABLE big (i int4, v vector(4000));");
$node->safe_psql("postgres", "INSERT INTO big SELECT i, " . random_array_sql(4000) . " FROM generate_series(1, 200) i;");
($ret, $stdout, $stderr) = $node->psql("postgres", "CREATE INDEX ON big USING hnsw (v vector_l2_ops);");
like($stderr, qr/column cannot have more than 2000 dimensions/);
($ret, $stdout, $stderr) = $node->psql("postgres", "CREATE INDEX ON big USING hnsw (v vector_l2_ops) WITH (rabitq_bits = 2, rabitq_keep_vectors = true);");
like($stderr, qr/column cannot have more than 2000 dimensions/);
$node->safe_psql("postgres", "CREATE INDEX ON big USING hnsw (v vector_l2_ops) WITH (rabitq_bits = 2);");
$node->safe_psql("postgres", "INSERT INTO big SELECT i, " . random_array_sql(4000) . " FROM generate_series(201, 300) i;");
$res = $node->safe_psql("postgres", qq(
	SET enable_seqscan = off;
	SELECT i FROM big ORDER BY v <-> (SELECT v FROM big WHERE i = 250) LIMIT 1;
));
is($res, 250, "4000 dimensions");

# Unsupported configurations
($ret, $stdout, $stderr) = $node->psql("postgres", "CREATE INDEX ON tst USING hnsw (v vector_l1_ops) WITH (rabitq_bits = 2);");
like($stderr, qr/RaBitQ only supports/);

$node->safe_psql("postgres", "CREATE TABLE huge (v vector(16000));");
($ret, $stdout, $stderr) = $node->psql("postgres", "CREATE INDEX ON huge USING hnsw (v vector_l2_ops) WITH (rabitq_bits = 8);");
like($stderr, qr/rabitq_bits is too large for 16000 dimensions/);

done_testing();
