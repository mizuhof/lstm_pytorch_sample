GPU=1,3

run:
	export CUDA_VISIBLE_DEVICES=$(GPU)
	python main.py --batch 8 --device_ids $(GPU)
