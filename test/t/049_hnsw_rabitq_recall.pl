use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

# Experimental RaBitQ distance estimation for HNSW scans (hnsw.rabitq_bits)
# Recall thresholds are set below values measured with 50 queries on this data

my $node;
my @queries = ();
my @expected;
my $limit = 10;
my $dim = 128;

sub test_recall
{
	my ($min, $operator, $bits, $ef_search) = @_;
	my $correct = 0;
	my $total = 0;

	my $settings = qq(
		SET enable_seqscan = off;
		SET hnsw.rabitq_bits = $bits;
		SET hnsw.ef_search = $ef_search;
	);

	for my $i (0 .. $#queries)
	{
		my $actual = $node->safe_psql("postgres", qq(
			$settings
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
	note("$operator bits=$bits ef_search=$ef_search recall=$recall");
	cmp_ok($recall, ">=", $min, "$operator bits=$bits ef_search=$ef_search");
}

sub test_order
{
	my ($operator, $bits) = @_;

	# Results must come back in exact distance order
	my $res = $node->safe_psql("postgres", qq(
		SET enable_seqscan = off;
		SET hnsw.rabitq_bits = $bits;
		SELECT bool_and(d >= prev) FROM (
			SELECT d, lag(d, 1, '-Infinity') OVER (ORDER BY n) AS prev FROM (
				SELECT v $operator '$queries[0]' AS d, row_number() OVER () AS n
				FROM (SELECT v FROM tst ORDER BY v $operator '$queries[0]' LIMIT 40) s
			) t
		) u;
	));
	is($res, "t", "$operator bits=$bits ordered");
}

sub random_vector
{
	return "[" . join(",", map { rand() - 0.5 } (1 .. $dim)) . "]";
}

# Initialize node
$node = PostgreSQL::Test::Cluster->new('node');
$node->init;
$node->start;

# Create table
$node->safe_psql("postgres", "CREATE EXTENSION vector;");
$node->safe_psql("postgres", "CREATE TABLE tst (i int4, v vector($dim));");
$node->safe_psql("postgres", qq(
	INSERT INTO tst SELECT i, ARRAY(SELECT random() - 0.5 + i * 0 FROM generate_series(1, $dim))::vector
	FROM generate_series(1, 5000) i;
));

# Generate queries
for (1 .. 20)
{
	push(@queries, random_vector());
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

		# Get exact results
		@expected = ();
		foreach (@queries)
		{
			my $res = $node->safe_psql("postgres", "SELECT i FROM tst ORDER BY v $operator '$_' LIMIT $limit;");
			push(@expected, $res);
		}

		$node->safe_psql("postgres", "CREATE INDEX idx ON tst USING hnsw (v $opclass);");

		# Uses executor recheck
		my $explain = $node->safe_psql("postgres", qq(
			SET enable_seqscan = off;
			SET hnsw.rabitq_bits = 4;
			EXPLAIN ANALYZE SELECT i FROM tst ORDER BY v $operator '$queries[0]' LIMIT $limit;
		));
		like($explain, qr/Index Scan/);

		# Baseline without RaBitQ (measured about 0.93)
		test_recall(0.85, $operator, 0, 100);

		# More bits and a larger rerank depth should increase recall
		# (measured about 0.72-0.77, 0.96-0.97, and 0.93-0.94)
		test_recall(0.60, $operator, 1, 100);
		test_recall(0.90, $operator, 1, 400);
		test_recall(0.85, $operator, 4, 100);
		test_recall(0.85, $operator, 8, 100);

		test_order($operator, 1);
		test_order($operator, 4);

		$node->safe_psql("postgres", "DROP INDEX idx;");
	}
}

# Unsupported configurations
$node->safe_psql("postgres", "CREATE INDEX idx ON tst USING hnsw (v halfvec_l1_ops);");
my ($ret, $stdout, $stderr) = $node->psql("postgres", qq(
	SET enable_seqscan = off;
	SET hnsw.rabitq_bits = 1;
	SELECT i FROM tst ORDER BY v <+> '$queries[0]' LIMIT $limit;
));
like($stderr, qr/hnsw.rabitq_bits only supports/);
$node->safe_psql("postgres", "DROP INDEX idx;");

$node->safe_psql("postgres", "CREATE INDEX idx ON tst USING hnsw (v halfvec_l2_ops);");
($ret, $stdout, $stderr) = $node->psql("postgres", qq(
	SET enable_seqscan = off;
	SET hnsw.rabitq_bits = 1;
	SET hnsw.iterative_scan = relaxed_order;
	SELECT i FROM tst ORDER BY v <-> '$queries[0]' LIMIT $limit;
));
like($stderr, qr/does not support iterative scans/);

# Fewer than 64 dimensions
$node->safe_psql("postgres", "CREATE TABLE small (v vector(3));");
$node->safe_psql("postgres", "INSERT INTO small SELECT ARRAY[random(), random(), random()] FROM generate_series(1, 100);");
$node->safe_psql("postgres", "CREATE INDEX ON small USING hnsw (v vector_l2_ops);");
($ret, $stdout, $stderr) = $node->psql("postgres", qq(
	SET enable_seqscan = off;
	SET hnsw.rabitq_bits = 1;
	SELECT v FROM small ORDER BY v <-> '[1,2,3]' LIMIT 5;
));
like($stderr, qr/requires at least 64 dimensions/);

done_testing();
