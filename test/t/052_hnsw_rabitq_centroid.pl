use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

# RaBitQ codes are relative to a centroid sampled from the table when the
# index is built. This data is far from the origin, so recall is very low
# (measured under 0.1) without a centroid.
# Recall thresholds are set below values measured with 50 queries on this data

my $node;
my @queries = ();
my @expected;
my $limit = 10;
my $dim = 128;
my $offset = 2;

sub test_recall
{
	my ($min, $max, $operator, $ef_search, $name) = @_;
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
	cmp_ok($recall, ">=", $min, "$name $operator ef_search=$ef_search min");
	cmp_ok($recall, "<=", $max, "$name $operator ef_search=$ef_search max");
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
	return "ARRAY(SELECT $offset + random() - 0.5 + i * 0 FROM generate_series(1, $n))";
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
	push(@queries, "[" . join(",", map { $offset + rand() - 0.5 } (1 .. $dim)) . "]");
}

my @operators = ("<->", "<=>");
my @opclasses = ("vector_l2_ops", "vector_cosine_ops");

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

		for my $options ("rabitq_bits = 2", "rabitq_bits = 2, rabitq_keep_vectors = true")
		{
			# Centroid from sample (measured about 0.92)
			$node->safe_psql("postgres", "CREATE INDEX idx ON tst USING hnsw (v $opclass) WITH ($options);");
			test_recall(0.80, 1, $operator, 100, "$type build ($options)");

			# Vacuum and inserts use the centroid from the index
			$node->safe_psql("postgres", "DELETE FROM tst WHERE i % 2 = 0;");
			$node->safe_psql("postgres", "VACUUM tst;");
			$node->safe_psql("postgres", "INSERT INTO tst SELECT * FROM tst2 WHERE i % 2 = 0;");
			test_recall(0.80, 1, $operator, 100, "$type vacuum ($options)");

			$node->safe_psql("postgres", "DROP INDEX idx;");
		}
	}
}

$node->safe_psql("postgres", "ALTER TABLE tst ALTER COLUMN v TYPE vector($dim);");
$node->safe_psql("postgres", "ALTER TABLE tst2 ALTER COLUMN v TYPE vector($dim);");
get_expected("<->");

# No centroid when the index is created on an empty table
$node->safe_psql("postgres", "TRUNCATE tst;");
$node->safe_psql("postgres", "CREATE INDEX idx ON tst USING hnsw (v vector_l2_ops) WITH (rabitq_bits = 2);");
$node->safe_psql("postgres", "INSERT INTO tst SELECT * FROM tst2;");
test_recall(0, 0.5, "<->", 100, "empty");

# Reindex calculates the centroid
$node->safe_psql("postgres", "REINDEX INDEX idx;");
test_recall(0.80, 1, "<->", 100, "reindex");
$node->safe_psql("postgres", "DROP INDEX idx;");

# Parallel builds share the centroid with workers (in memory and on disk)
for my $mem ("64MB", "2MB")
{
	my ($ret, $stdout, $stderr) = $node->psql("postgres", qq(
		ALTER TABLE tst SET (parallel_workers = 2);
		SET client_min_messages = DEBUG;
		SET min_parallel_table_scan_size = 1;
		SET maintenance_work_mem = '$mem';
		CREATE INDEX idx ON tst USING hnsw (v vector_l2_ops) WITH (rabitq_bits = 2);
		ALTER TABLE tst RESET (parallel_workers);
	));
	is($ret, 0, $stderr);
	like($stderr, qr/using \d+ parallel workers/);
	if ($mem eq "2MB")
	{
		like($stderr, qr/hnsw graph no longer fits into maintenance_work_mem/);
	}
	test_recall(0.80, 1, "<->", 100, "parallel $mem");

	if ($mem eq "2MB")
	{
		# Crash recovery
		$node->safe_psql("postgres", "INSERT INTO tst SELECT i, " . random_array_sql($dim) . " FROM generate_series(5001, 5500) i;");
		$node->stop('immediate');
		$node->start;
		get_expected("<->");
		test_recall(0.80, 1, "<->", 100, "recovery");
	}

	$node->safe_psql("postgres", "DROP INDEX idx;");
}

# Partial and expression indexes
$node->safe_psql("postgres", "CREATE INDEX idx ON tst USING hnsw ((v::halfvec($dim)) halfvec_l2_ops) WITH (rabitq_bits = 2) WHERE i > 0;");
my $res = $node->safe_psql("postgres", qq(
	SET enable_seqscan = off;
	SET hnsw.ef_search = 100;
	SELECT i FROM tst WHERE i > 0 ORDER BY v::halfvec($dim) <-> (SELECT v::halfvec($dim) FROM tst WHERE i = 123) LIMIT 1;
));
is($res, 123, "expression index");
$node->safe_psql("postgres", "DROP INDEX idx;");

# Centroid spans multiple pages
$node->safe_psql("postgres", "CREATE TABLE big (i int4, v vector(4000));");
$node->safe_psql("postgres", "INSERT INTO big SELECT i, " . random_array_sql(4000) . " FROM generate_series(1, 300) i;");
$node->safe_psql("postgres", "CREATE INDEX ON big USING hnsw (v vector_l2_ops) WITH (rabitq_bits = 2);");
$node->safe_psql("postgres", "INSERT INTO big SELECT i, " . random_array_sql(4000) . " FROM generate_series(301, 400) i;");
my $found = 0;
for my $i (50, 150, 250, 350)
{
	$res = $node->safe_psql("postgres", qq(
		SET enable_seqscan = off;
		SET hnsw.ef_search = 100;
		SELECT i FROM big ORDER BY v <-> (SELECT v FROM big WHERE i = $i) LIMIT 1;
	));
	$found++ if $res eq $i;
}
is($found, 4, "4000 dimensions");

done_testing();
