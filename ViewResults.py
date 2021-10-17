#@title View Results WESAD
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd

from WesadSimulation import ScenarioContainerWesad
from WesadClasses import *

sc = ScenarioContainerWesad()
view_scenario_index=0
global_scenario = sc.global_scenarios[0]
print(global_scenario)

data_key = global_scenario['data']
target_key = global_scenario['target']
fl_key = global_scenario['fl']
defense_key = global_scenario['defense']
attack_key = global_scenario['attack']

fl_results_file_name = 'results/fl_results_{}.pkl'.format('_'.join([data_key, target_key, fl_key, defense_key]))
privacy_results_filename = "results/privacy_results_" + "_".join([data_key, target_key, defense_key, attack_key]) + ".pkl"
centralized_results_file_name = 'results/cen_results_{}.pkl'.format('_'.join([data_key, target_key]))

try:
  with open(fl_results_file_name, 'rb') as file:
    fl_results = pkl.load(file)
  with open(centralized_results_file_name, 'rb') as file:
    central_results = pkl.load(file)
  fig, ax = plt.subplots(1, 2, figsize=(15, 6))
  ax[0].set_title("Target Classifier Loss ({})".format(view_scenario_index), fontsize=12)
  ax[0].set_ylabel("Loss")
  ax[0].set_xlabel("Gradient Updates")
  ax[0].set_xlim(0, fl_results['scenarios']['fl'][FL_PARAMS.GRADIENT_UPDATES])

  temp = pd.Series(central_results['results']['loss']).rolling(3).agg('mean')
  ax[0].plot(temp, label="Centralized")

  for i in range(len(fl_results['results']['defense_params'])):
    temp = pd.Series(fl_results['results']['loss'][i]).rolling(3).agg('mean')
    ax[0].plot(temp, label=fl_results['results']['defense_params'][i])
  ax[0].legend()

  acc_metric = "Accuracy" if fl_results['scenarios']['data'][DATA_PARAMS.IS_BINARY] else "R^2"
  ax[1].set_title("Target Classifier {} ({})".format(acc_metric, view_scenario_index), fontsize=12)
  ax[1].set_ylabel(acc_metric)
  ax[1].set_xlabel("Gradient Updates")
  ax[1].set_xlim(0, fl_results['scenarios']['fl'][FL_PARAMS.GRADIENT_UPDATES])

  temp = pd.Series(central_results['results']['accuracy']).rolling(3).agg('mean')
  ax[1].plot(temp, label="Centralized")

  for i in range(len(fl_results['results']['defense_params'])):
    temp = pd.Series(fl_results['results']['accuracy'][i]).rolling(3).agg('mean')
    ax[1].plot(temp, label=fl_results['results']['defense_params'][i])
  ax[1].legend()

  plt.savefig("out_fig.png")

except FileNotFoundError:
  print("No FL results for " + str(view_scenario_index) + " found")
  print(fl_results_file_name)

try:
  with open(privacy_results_filename, 'rb') as file:
    privacy_results = pkl.load(file)
  print(privacy_results['scenarios']['defense'][DEFENSE_PARAMS.DEFENSE_STRATEGY_LIST])
  for i in range(len(privacy_results['results']['conditions'])):
    average = sum([sum(attack_seg) for attack_seg in privacy_results['results']['measures'][i]])/(privacy_results['scenarios']['attack'][ATTACK_PARAMS.SEGMENTS] * privacy_results['scenarios']['attack'][ATTACK_PARAMS.TEST_REPEATS])
    print("Condition {} Accuracy {}".format(privacy_results['results']['conditions'][i], average))

except FileNotFoundError:
  print("No privacy results for " + str(view_scenario_index) + " found")
  print(privacy_results_filename)