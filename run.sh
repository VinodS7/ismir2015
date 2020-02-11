num_models=5

for iteration in $(seq 1 $num_models)
do
	mkdir model_data/model$iteration
done

# code to read arguments to the bash script
# getopts works with either single word arguments -s or there is something like --long, something like -se 1 not works

echo "Script Starts!"

for iteration in $(seq 1 $num_models)
do

	echo "---------Training Model$iteration---------"
	python experiments/train.py model_data/model$iteration/ --cache-spectra experiments/mel_data --var initial_eta=0.01 --var momentum=0.95 --no-augment --validate

	echo "***********Prediction with Model$iteration***********"
	echo "predict mode - one neuron model"		
	python experiments/predict.py model_data/model$iteration/model.pth model_data/model$iteration/jamendo_pred.npz --cache-spectra experiments/mel_data/

    echo "---------Evaluation with Model$iteration-------------"
    echo "evaluate mode"
    
    python experiments/eval.py model_data/model$iteration/jamendo_pred.npz 
done


