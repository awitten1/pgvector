use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

# Experimental RaBitQ codes stored with vectors in HNSW indexes
# (rabitq_bits and rabitq_keep_vectors index options)
# Recall thresholds are set below values measured with 50 queries on this data

my $node;
my @queries = ();
my @expected;
my $limit = 10;
my $dim = 128;

sub test_recall
{
	my ($min, $operator, $bits, $ef_search, $name) = @_;
	my $correct = 0;
	my $total = 0;

	for my $i (0 .. $#queries)
	{
		my $actual = $node->safe_psql("postgres", qq(
			SET enable_seqscan = off;
			SET hnsw.rabitq_bits = $bits;
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
	note("$name $operator bits=$bits ef_search=$ef_search recall=$recall");
	cmp_ok($recall, ">=", $min, "$name $operator bits=$bits ef_search=$ef_search");
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

my $insert_sql = "SELECT i, ARRAY(SELECT random() - 0.5 + i * 0 FROM generate_series(1, $dim))";

# Initialize node
$node = PostgreSQL::Test::Cluster->new('node');
$node->init;
$node->start;

# Create table
$node->safe_psql("postgres", "CREATE EXTENSION vector;");
$node->safe_psql("postgres", "CREATE TABLE tst (i int4, v vector($dim));");
$node->safe_psql("postgres", "INSERT INTO tst $insert_sql FROM generate_series(1, 5000) i;");

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
	}

	for my $i (0 .. $#operators)
	{
		my $operator = $operators[$i];
		my $opclass = $opclasses[$i];
		$opclass =~ s/^vector/halfvec/ if $type eq "halfvec";

		get_expected($operator);

		$node->safe_psql("postgres", "CREATE INDEX idx ON tst USING hnsw (v $opclass) WITH (rabitq_bits = 2, rabitq_keep_vectors = true);");

		# Stored codes are used by default (measured about 0.91-0.94 and 0.98)
		test_recall(0.80, $operator, -1, 100, "$type stored");
		test_recall(0.93, $operator, -1, 200, "$type stored");
		test_order($operator);

		# Exact distances can still be used (measured about 0.93-0.95)
		test_recall(0.85, $operator, 0, 100, "$type exact");

		# Other bits are quantized on the fly
		test_recall(0.85, $operator, 4, 100, "$type on the fly");

		$node->safe_psql("postgres", "DROP INDEX idx;");
	}
}

$node->safe_psql("postgres", "ALTER TABLE tst ALTER COLUMN v TYPE vector($dim);");

# Codes take space
$node->safe_psql("postgres", "CREATE INDEX plain ON tst USING hnsw (v vector_l2_ops);");
$node->safe_psql("postgres", "CREATE INDEX idx ON tst USING hnsw (v vector_l2_ops) WITH (rabitq_bits = 8, rabitq_keep_vectors = true);");
my $plain_size = $node->safe_psql("postgres", "SELECT pg_relation_size('plain');");
my $size = $node->safe_psql("postgres", "SELECT pg_relation_size('idx');");
cmp_ok($size, ">", $plain_size, "index with codes is larger");
$node->safe_psql("postgres", "DROP INDEX plain;");
$node->safe_psql("postgres", "DROP INDEX idx;");

# Inserts
$node->safe_psql("postgres", "CREATE TABLE tst2 AS SELECT * FROM tst;");
$node->safe_psql("postgres", "TRUNCATE tst;");
$node->safe_psql("postgres", "CREATE INDEX idx ON tst USING hnsw (v vector_l2_ops) WITH (rabitq_bits = 2, rabitq_keep_vectors = true);");
$node->safe_psql("postgres", "INSERT INTO tst SELECT * FROM tst2;");
get_expected("<->");
test_recall(0.80, "<->", -1, 100, "inserts");

# Vacuum and reuse of space
$node->safe_psql("postgres", "DELETE FROM tst WHERE i % 2 = 0;");
$node->safe_psql("postgres", "VACUUM tst;");
$node->safe_psql("postgres", "INSERT INTO tst SELECT * FROM tst2 WHERE i % 2 = 0;");
test_recall(0.80, "<->", -1, 100, "vacuum");
test_order("<->");
$node->safe_psql("postgres", "DROP INDEX idx;");

# On-disk and parallel builds
my ($ret, $stdout, $stderr) = $node->psql("postgres", qq(
	ALTER TABLE tst SET (parallel_workers = 2);
	SET client_min_messages = DEBUG;
	SET maintenance_work_mem = '2MB';
	CREATE INDEX idx ON tst USING hnsw (v vector_l2_ops) WITH (rabitq_bits = 2, rabitq_keep_vectors = true);
	ALTER TABLE tst RESET (parallel_workers);
));
is($ret, 0, $stderr);
like($stderr, qr/using \d+ parallel workers/);
like($stderr, qr/hnsw graph no longer fits into maintenance_work_mem/);
test_recall(0.80, "<->", -1, 100, "on-disk build");

# Crash recovery
$node->safe_psql("postgres", "INSERT INTO tst $insert_sql FROM generate_series(5001, 5500) i;");
$node->stop('immediate');
$node->start;
get_expected("<->");
test_recall(0.80, "<->", -1, 100, "recovery");

# Iterative scans fall back to exact distances
my $count = $node->safe_psql("postgres", qq(
	SET enable_seqscan = off;
	SET hnsw.iterative_scan = relaxed_order;
	SELECT COUNT(*) FROM (SELECT i FROM tst WHERE i % 100 = 0 ORDER BY v <-> '$queries[0]' LIMIT 50) s;
));
is($count, 50, "iterative scan");
$node->safe_psql("postgres", "DROP INDEX idx;");

# Unsupported configurations
($ret, $stdout, $stderr) = $node->psql("postgres", "CREATE INDEX ON tst USING hnsw (v vector_l1_ops) WITH (rabitq_bits = 2, rabitq_keep_vectors = true);");
like($stderr, qr/RaBitQ only supports/);

$node->safe_psql("postgres", "CREATE TABLE small (v vector(3));");
($ret, $stdout, $stderr) = $node->psql("postgres", "CREATE INDEX ON small USING hnsw (v vector_l2_ops) WITH (rabitq_bits = 2, rabitq_keep_vectors = true);");
like($stderr, qr/RaBitQ requires at least 64 dimensions/);

$node->safe_psql("postgres", "CREATE TABLE big (v vector(2000));");
($ret, $stdout, $stderr) = $node->psql("postgres", "CREATE INDEX ON big USING hnsw (v vector_l2_ops) WITH (rabitq_bits = 4, rabitq_keep_vectors = true);");
like($stderr, qr/rabitq_bits is too large for 2000 dimensions/);

($ret, $stdout, $stderr) = $node->psql("postgres", "CREATE INDEX ON big USING hnsw (v vector_l2_ops) WITH (rabitq_bits = 10, rabitq_keep_vectors = true);");
like($stderr, qr/value 10 out of bounds for option "rabitq_bits"/);

done_testing();
