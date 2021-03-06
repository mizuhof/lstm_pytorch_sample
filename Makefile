GPU=1,3
BATCH=4096
EPOCH=10000
LOG=log
EVAL_FREQ=100

run:
	export CUDA_VISIBLE_DEVICES=$(GPU)
	python main.py \
		--batch $(BATCH) --device_ids $(GPU) \
		--epochs $(EPOCH) --log_path $(LOG) \
		--eval_freq $(EVAL_FREQ)
