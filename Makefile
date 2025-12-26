SETTINGS ?= SETTINGS.json

.PHONY: prepare-data train predict

prepare-data:
	python prepare_data.py --settings $(SETTINGS)

train:
	python train.py --settings $(SETTINGS)

predict:
	python predict.py --settings $(SETTINGS)
