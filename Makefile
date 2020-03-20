install:
	python setup.py

black:
	black -S ./pyspikelib ./examples
