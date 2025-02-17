from generic_training import train
import optuna
from functools import partial


def objective(trial):

    epoch = 5

    label_smoothing_rate = trial.suggest_float("label_smoothing_rate", -10, 0.3)

    final_val = train(epoch, label_smoothing_rate)
    f1 = open('Results_Recall.txt', 'a+')
    f1.write("smoothing factor is: " + str(trial) + str(label_smoothing_rate) + '\n')
    f1.close()

    return final_val

number_of_runs = 50

study = optuna.create_study(direction="maximize")
objective_with_params = partial(objective)
study.optimize(objective_with_params, n_trials=number_of_runs)

best_trial = study.best_trial
print("++++++++++++Best results after trials with Bayesian optimization++++++++++++", best_trial)


#final_val = train(8, 0.0)

