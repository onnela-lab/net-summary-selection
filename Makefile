.PHONY : sync tests

requirements.txt : requirements.in
	pip-compile -v

sync : requirements.txt
	pip-sync
