#@title Run Simulation (WESAD)
import sys
from WesadClasses import *
from WesadFunctions import *

class ScenarioContainerWesad:
    def __init__(self):
        self.data_scenarios = {
            'data_wesad': {
                DATA_PARAMS.IS_BINARY: True,
                DATA_PARAMS.MICE_ITERATIONS: -1,
                DATA_PARAMS.MIN_OBS_PER_CLIENT: -1,
                DATA_PARAMS.NP_SEED: 42
            }
        }
        self.target_scenarios = {
            'target_wesad':{
                TARGET_PARAMS.TORCH_SEED: 67,
                TARGET_PARAMS.EPOCHS: 5,
                TARGET_PARAMS.BATCH_SIZE: 256,
                TARGET_PARAMS.LEARNING_RATE: 0.001,
                TARGET_PARAMS.FIRST_LAYER_FEATURES: 12,
                TARGET_PARAMS.LAYER_SCALINGS: [12/10, 10/8],
                TARGET_PARAMS.DROPOUT: 0.2
            }
        }
        self.fl_scenarios = {
            'fl_wesad': {
                FL_PARAMS.TORCH_SEED: 2427,
                FL_PARAMS.NP_SEED: 42,
                FL_PARAMS.GRADIENT_UPDATES: 200,
                FL_PARAMS.CLIENT_PROP_PER_ROUND: 0.3,
                FL_PARAMS.LEARNING_RATE: 0.001,
                FL_PARAMS.DATA_MULTIPLIER: 1
            }
        }
        self.defense_scenarios = {
            'defense_unscaled_nc': {
                DEFENSE_PARAMS.TORCH_SEED: 242,
                DEFENSE_PARAMS.DEFENSE_STRATEGY_LIST: [DEFENSE_STRATEGY.NOISE_CLIP],
                DEFENSE_PARAMS.DEFENSE_STRATEGY_PARAMS: [
                    [
                        (0, 1),
                        (0.001, 1),
                        (0.002, 1),
                        (0.005, 1),
                        (0.01, 1),
                        (0.02, 1),
                        (0.03, 1),
                        (0.04, 1)
                    ]                      
                ],
                DEFENSE_PARAMS.SCALE_GRADIENT_BY_SAMPLE_SIZE: False,
                DEFENSE_PARAMS.LINF_CLIP: False,
                DEFENSE_PARAMS.MAINTAIN_GRADIENT_SIGN: False,
                DEFENSE_PARAMS.CENSOR_SINGLE_TARGET: False
            }
        }
        self.attack_scenarios = {
            'attack_kckn_yes_normal': {
                ATTACK_PARAMS.TORCH_SEED: 242,
                ATTACK_PARAMS.NP_SEED: 42,
                ATTACK_PARAMS.DATA_ACCESS_MULTI: 1,
                ATTACK_PARAMS.TEST_REPEATS: 1,
                ATTACK_PARAMS.CLIP_IS_KNOWN: True,
                ATTACK_PARAMS.NOISE_IS_KNOWN: True,
                ATTACK_PARAMS.NORMALIZE_GRADIENTS: True,
                ATTACK_PARAMS.SEGMENTS: 4,
                ATTACK_PARAMS.FOLDS: 3,
                ATTACK_PARAMS.MAX_TRAIN_EPOCHS: 800, 
                ATTACK_PARAMS.BATCH_SIZE: 256,
                ATTACK_PARAMS.LEARNING_RATE: 0.001,
                ATTACK_PARAMS.FIRST_LAYER_FEATURES: 12,
                ATTACK_PARAMS.LAYER_SCALINGS: [12/10, 10/8],
                ATTACK_PARAMS.DROPOUT: 0.2,
                ATTACK_PARAMS.WEIGHT_DECAY: 0.1,
                ATTACK_PARAMS.TRAIN_SAMPLE_MULTI: 5,
                ATTACK_PARAMS.TEST_SAMPLE_MULTI: 10
            }
        }
        self.global_scenarios = [
            {
                'data': 'data_wesad',
                'target': 'target_wesad',
                'fl': 'fl_wesad',
                'defense': 'defense_unscaled_nc',
                'attack': 'attack_kckn_yes_normal'
            }
        ]

if __name__ == '__main__':
    DEVICE = my_device()

    sc = ScenarioContainerWesad()
    data_prop = float(sys.argv[2])
    pieces = int(sys.argv[1])

    global_scenario = sc.global_scenarios[0]
    print(global_scenario)

    data_key = global_scenario['data']
    data_scenario = sc.data_scenarios[data_key]
    target_key = global_scenario['target']
    target_scenario = sc.target_scenarios[target_key]
    fl_key = global_scenario['fl']
    fl_scenario = sc.fl_scenarios[fl_key]
    defense_key = global_scenario['defense']
    defense_scenario = sc.defense_scenarios[defense_key]
    attack_key = global_scenario['attack']
    attack_scenario = sc.attack_scenarios[attack_key]


    fl_results_file_name = 'results/fl_results_{}.pkl'.format('_'.join([data_key, target_key, fl_key, defense_key]))
    privacy_results_filename = "results/privacy_results_" + "_".join([data_key, target_key, defense_key, attack_key]) + ".pkl"
    centralized_results_file_name = 'results/cen_results_{}.pkl'.format('_'.join([data_key, target_key]))

    try:
        with open(fl_results_file_name, 'rb') as file:
            pass
        with open(privacy_results_filename, 'rb') as file:
            pass 
        with open(centralized_results_file_name, 'rb') as file:
            pass
        print("Central, FL, and Privacy results exist, skipping")
        sys.exit()
    except FileNotFoundError:
        print("Central, FL, or Privacy results need to be generated")

    # Generate Data
    data_filename = "temp_files/dataset_" + data_key + ".pkl"

    try:
        with open(data_filename, 'rb') as file:
            final_df, unscaled_df = pkl.load(file)
            print("Data Loaded")
    except FileNotFoundError:
        print("Data Preprocessing Started")
        try:
            with open("src/client_split/split.done", 'r') as file:
                pass 
        except FileNotFoundError:
            split_files(pieces)
        final_df, unscaled_df = wesad_preprocess(data_prop, data_scenario[DATA_PARAMS.NP_SEED])
        with open(data_filename, 'wb') as file:
            pkl.dump((final_df, unscaled_df), file)
        print("Data Preprocessing Finished")

    # Generate Target Model
    all_train_predictors, all_train_response, all_train_weights, client_data_dict = data_loading_wesad(final_df)
    target_model = WESADNN(
        data_scenario[DATA_PARAMS.IS_BINARY],
        all_train_predictors.shape[-1],
        target_scenario[TARGET_PARAMS.FIRST_LAYER_FEATURES],
        target_scenario[TARGET_PARAMS.LAYER_SCALINGS],
        target_scenario[TARGET_PARAMS.DROPOUT]
    ).to(DEVICE)
    sd_path = "temp_files/targetmodel_" + data_key + "_" + target_key + ".sd"

    try:
        target_model.load_state_dict(torch.load(sd_path))
        print("Target Loaded")
    except FileNotFoundError:
        print("Target Training Started")
        train_centralized_model(
            target_model=target_model,
            data_scenario=data_scenario,
            target_scenario=target_scenario,
            train_predictors=all_train_predictors,
            train_response=all_train_response,
            train_weights=all_train_weights,
            sd_path=sd_path
        )
        print("Target Training Finished")

    # Centralized Convergence
    try:
        with open(centralized_results_file_name, 'rb') as file:
            print("Centralized Results Exist")
    except FileNotFoundError:
        print("Generating Centralized Results")
        run_centralized_convergence_test(
            data_scenario,
            target_scenario,
            all_train_predictors,
            all_train_response,
            all_train_weights,
            centralized_results_file_name,
            is_wesad=True
        )
        print("Centralized Results Finished")

    # FL Tests
    ### Baseline No PP
    no_pp_params = [defense_scenario[DEFENSE_PARAMS.SCALE_GRADIENT_BY_SAMPLE_SIZE]]
    fl_no_pp_filename = "temp_files/fl_no_pp_{}_params_{}.pkl".format('_'.join([data_key, fl_key, target_key]), '_'.join([str(p) for p in no_pp_params]))

    try:
        with open(fl_no_pp_filename, 'rb') as file:
            fl_no_pp_results = pkl.load(file)
            print("FL No PP Results Loaded")
    except FileNotFoundError:
        print("FL No PP Results Generating")
        fl_no_pp_results = run_no_protection_convergence_test(data_scenario, target_scenario, fl_scenario, defense_scenario, all_train_predictors, all_train_response, all_train_weights, client_data_dict, is_wesad=True)
        with open(fl_no_pp_filename, 'wb') as file:
            pkl.dump(fl_no_pp_results, file)
        print("FL No PP Results Finished")

    ## Other Tests
    try:
        with open(fl_results_file_name, 'rb') as file:
            fl_results = pkl.load(file)
            print("FL Results Exist")
    except FileNotFoundError:
        print("FL Results Generating")
        run_federated_convergence_tests(
            fl_no_pp_results,
            data_scenario,
            target_scenario,
            fl_scenario,
            defense_scenario,
            all_train_predictors,
            all_train_response,
            all_train_weights,
            client_data_dict,
            fl_results_file_name,
            is_wesad=True
        )
        print("FL Results Finished")

    # privacy test
    no_pp_params = [
        defense_scenario[DEFENSE_PARAMS.SCALE_GRADIENT_BY_SAMPLE_SIZE]
    ]
    privacy_no_pp_filename = "temp_files/privacy_no_pp_{}_params_{}.pkl".format('_'.join([data_key, target_key, attack_key]), '_'.join([str(p) for p in no_pp_params]))

    try:
        with open(privacy_no_pp_filename, 'rb') as file:
            privacy_no_pp_results = pkl.load(file)
            print("Privacy No PP Results Loaded")
    except FileNotFoundError:
        print("Privacy No PP Results Generating")
        privacy_no_pp_results = run_no_protection_privacy_test(target_model, client_data_dict, unscaled_df, defense_scenario, attack_scenario, 9999, is_wesad=True)
        with open(privacy_no_pp_filename, 'wb') as file:
            pkl.dump(privacy_no_pp_results, file)
        print("Privacy No PP Results Finished")

    try:
        with open(privacy_results_filename, 'rb') as file:
            privacy_results = pkl.load(file)
            print("Privacy Results Exist")
    except FileNotFoundError:
        print("Privacy Results Generating")
        run_privacy_tests(privacy_no_pp_results, data_scenario, target_scenario, defense_scenario, attack_scenario, target_model, unscaled_df, client_data_dict, privacy_results_filename, 9999, is_wesad=True)
        print("Privacy Results Finished")