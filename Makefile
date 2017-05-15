inplace:
	python setup.py build_ext --inplace

deb:
	python setup.py --command-packages=stdeb.command sdist_dsc bdist_deb

clean:
	rm deb_dist pyueye*.tar.gz build dist MANIFEST -rf
	python setup.py clean
