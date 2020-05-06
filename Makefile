install:
	pip install -r requirements.txt

black:
	black -S ./pyspikelib ./examples
