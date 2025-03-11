install:
	pip install -e .[dev]

test:
	pytest us_imputation_benchmarking/tests -s

format:
	black . -l 79
