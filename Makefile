init: init_classifiers
	pip install -r requirements.txt

init_classifiers:
	wget -O './data/classifiers/shape_predictor_68_face_landmarks.dat' \
		'https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat?raw=true'

test:
	py.test tests

.PHONY: init test
