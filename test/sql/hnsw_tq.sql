SET enable_seqscan = off;

-- L2 with exact re-rank

CREATE TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (quantizer = tq4, rerank = exact);

INSERT INTO t (val) VALUES ('[1,2,4]');

SELECT * FROM t ORDER BY val <-> '[3,3,3]';
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <-> (SELECT NULL::vector)) t2;
SELECT COUNT(*) FROM t;

TRUNCATE t;
SELECT * FROM t ORDER BY val <-> '[3,3,3]';

DROP TABLE t;

-- L2 with nested re-rank codes

CREATE TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (quantizer = tq4, rerank = ntq12);

INSERT INTO t (val) VALUES ('[1,2,4]');

SELECT * FROM t ORDER BY val <-> '[3,3,3]';

DROP TABLE t;

-- inner product

CREATE TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
CREATE INDEX ON t USING hnsw (val vector_ip_ops) WITH (quantizer = tq4, rerank = ntq10);

INSERT INTO t (val) VALUES ('[1,2,4]');

SELECT * FROM t ORDER BY val <#> '[3,3,3]';
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <#> (SELECT NULL::vector)) t2;

DROP TABLE t;

-- cosine

CREATE TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
CREATE INDEX ON t USING hnsw (val vector_cosine_ops) WITH (quantizer = tq4, rerank = exact);

INSERT INTO t (val) VALUES ('[1,2,4]');

SELECT * FROM t ORDER BY val <=> '[3,3,3]';
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <=> '[0,0,0]') t2;

DROP TABLE t;

-- recall, inserts, and vacuum

SELECT setseed(0.5);
CREATE TABLE t (id int, val vector(64));
INSERT INTO t SELECT i, ARRAY(SELECT random()::real FROM generate_series(1, 64) g WHERE g > -i)::vector FROM generate_series(1, 1000) i;
CREATE INDEX t_idx ON t USING hnsw (val vector_l2_ops) WITH (quantizer = tq4, rerank = ntq12);
INSERT INTO t SELECT i, ARRAY(SELECT random()::real FROM generate_series(1, 64) g WHERE g > -i)::vector FROM generate_series(1001, 2000) i;

SET hnsw.ef_search = 100;

CREATE FUNCTION tq_recall() RETURNS float AS $$
	SELECT SUM((
		SELECT COUNT(*) FROM (SELECT id FROM t ORDER BY val <-> q.val LIMIT 10) a
		WHERE a.id IN (SELECT id FROM t ORDER BY (val <-> q.val) + 0 LIMIT 10)
	))::float / (10 * COUNT(*))
	FROM (SELECT val FROM t WHERE id % 100 = 0) q
$$ LANGUAGE SQL;

SELECT tq_recall() >= 0.9 AS recall_ok;

CREATE TABLE deleted AS SELECT * FROM t WHERE id % 2 = 0;
DELETE FROM t WHERE id % 2 = 0;
VACUUM t;
SELECT pg_relation_size('t_idx') AS size_before \gset
INSERT INTO t SELECT * FROM deleted;
SELECT pg_relation_size('t_idx') = :size_before AS slots_reused;
SELECT COUNT(*) FROM t;
SELECT tq_recall() >= 0.9 AS recall_ok;

-- iterative scans are not supported yet
SET hnsw.iterative_scan = relaxed_order;
SELECT id FROM t ORDER BY val <-> (SELECT val FROM t WHERE id = 1) LIMIT 1;
RESET hnsw.iterative_scan;

DROP FUNCTION tq_recall();
DROP TABLE t, deleted;

-- options and unsupported opclasses and types

CREATE TABLE t (val vector(3));
CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (quantizer = tq8);
CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (quantizer = tq4, rerank = ntq16);
CREATE INDEX ON t USING hnsw (val vector_l1_ops) WITH (quantizer = tq4);
DROP TABLE t;

CREATE TABLE t (val halfvec(3));
CREATE INDEX ON t USING hnsw (val halfvec_l2_ops) WITH (quantizer = tq4);
DROP TABLE t;
