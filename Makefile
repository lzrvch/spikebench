install:
	pip install .

black:
	black -S ./pyspikelib ./examples
