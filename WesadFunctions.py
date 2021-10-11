import pickle as pkl
import pandas as pd
import torch 
import pandas as pd
import numpy as np
import pickle as pkl
import torch.nn.functional as F
import glob

from sklearn.preprocessing import StandardScaler

from WesadClasses import *

def get_client_ids():
    client_ids_numbers = list(range(18))
    client_ids_numbers.pop(12)
    client_ids_numbers.pop(0)
    client_ids_numbers.pop(0)
    return list(map(lambda x: 'S' + str(x), client_ids_numbers))

def split_files():
    global_window_size = 21000
    global_pieces = 100

    for client_id in get_client_ids():
        print("Processing {}".format(client_id))
        client_df = pd.DataFrame()

        with open(r'src\WESAD\{}\{}.pkl'.format(client_id, client_id), 'rb') as file:
            data = pkl.load(file, encoding='latin1')

        for mode in ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']:
            if mode == 'ACC':
                i = 0
                for axis in ["_X", "_Y", "_Z"]:
                    client_df[mode+axis] = data['signal']['chest'][mode][:, i]
                    i += 1
            else:
                client_df[mode] = data['signal']['chest'][mode]

        client_df['labels'] = data['label']
        client_df = client_df[client_df['labels'].isin([1, 2, 3])]
        client_agg_df = None
        for label in (1, 2, 3):
            if client_agg_df is None:
                client_agg_df = client_df[client_df["labels"]==label].drop(columns='labels').rolling(global_window_size).agg(['mean', 'std', 'min', 'max'])
                client_agg_df = client_agg_df[~client_agg_df.isna().any(axis=1)]
                client_agg_df['labels'] = label
                client_agg_df = [client_agg_df.iloc[int(i * len(client_agg_df)/global_pieces):int((i+1) * len(client_agg_df)/global_pieces),:] for i in range(global_pieces)]
            else:
                temp_df = client_df[client_df["labels"]==label].drop(columns='labels').rolling(global_window_size).agg(['mean', 'std', 'min', 'max'])
                temp_df = temp_df[~temp_df.isna().any(axis=1)]
                temp_df['labels'] = label
                for i in range(global_pieces):
                    client_agg_df[i] = client_agg_df[i].append(temp_df.iloc[int(i * len(temp_df)/global_pieces):int((i+1) * len(temp_df)/global_pieces),:])

        for sub_df in client_agg_df:
            sub_df.columns = [' '.join(col).strip() for col in sub_df.columns.values]

        for i in range(global_pieces):
            with open("src/client_split/{}_{}.pkl".format(client_id, i), 'wb') as file:
                pkl.dump(client_agg_df[i], file)

def wesad_preprocess(data_proportion, np_seed):
    client_id = 0
    unscaled_df = None
    np.random.seed(np_seed)
    for filepath in glob.iglob('src/client_split/*.pkl'):
        with open(filepath, 'rb') as file:
            temp_df = pkl.load(file)
            temp_df['client_id'] = client_id 

        
        if unscaled_df is None:
            unscaled_df = temp_df 
        else:
            if np.random.random() < data_proportion:
                unscaled_df = unscaled_df.append(temp_df)
                client_id += 1

    scaled_df = unscaled_df.copy().drop(columns=["client_id", "labels"])
    scaler = StandardScaler()
    scaler.fit(scaled_df.loc[:, :].to_numpy())
    scaled_df.loc[:, :] = scaler.transform(scaled_df.loc[:, :].to_numpy())
    scaled_df["client_id"] = unscaled_df["client_id"]
    scaled_df["labels"] = unscaled_df["labels"]
    return scaled_df, unscaled_df

def data_loading_wesad(final_df):
    DEVICE = my_device()
    all_train_response_df = final_df['labels'] - 1

    all_train_response = torch.tensor(all_train_response_df.to_numpy(), dtype=torch.long, device=DEVICE)
    all_train_predictors = torch.tensor(final_df.drop(columns=['client_id', 'labels']).to_numpy(), dtype=torch.float, device=DEVICE)
    all_train_weights = None

    client_data_dict = {}
    for client_id in final_df['client_id'].unique():
        client_df = final_df.query('client_id == {}'.format(client_id))
        train_response_df = client_df['labels'] - 1
        client_weights = None
        
        client_response = torch.tensor(train_response_df.to_numpy(), dtype=torch.long, device=DEVICE)
        client_predictors = torch.tensor(client_df.drop(columns=['client_id', 'labels']).to_numpy(), dtype=torch.float, device=DEVICE)
        client_data_dict[client_id] = { 'data': client_predictors, 'labels':client_response, 'weights': client_weights}
    
    return all_train_predictors, all_train_response, all_train_weights, client_data_dict

def train_centralized_model(target_model, data_scenario, target_scenario, train_predictors, train_response, train_weights, sd_path):
    DEVICE = my_device()
    is_binary = data_scenario[DATA_PARAMS.IS_BINARY] 
    torch.manual_seed(target_scenario[TARGET_PARAMS.TORCH_SEED])

    train_ds = IHSDataset(train_predictors, train_response, train_weights)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=target_scenario[TARGET_PARAMS.BATCH_SIZE], shuffle=True)

    target_model.create_optimizer(target_scenario[TARGET_PARAMS.LEARNING_RATE])
    for i in range(target_scenario[TARGET_PARAMS.EPOCHS]):
        target_model.train()
        for batch in train_loader:
            target_model.train_batch(batch)

        train_loss, train_acc, _ = target_model.compute_loss_acc(train_predictors, train_response, weights=None)
        print("{} Train Loss: {} | Train {}: {}".format(i+1, train_loss, 'Acc' if is_binary else 'R2', train_acc))

    torch.save(target_model.state_dict(), sd_path)


def run_centralized_convergence_test(data_scenario, target_scenario, all_train_predictors, all_train_response, all_train_weights, results_path, is_wesad=False):
    DEVICE = my_device()
    torch.manual_seed(target_scenario[TARGET_PARAMS.TORCH_SEED])

    train_ds = IHSDataset(all_train_predictors, all_train_response, all_train_weights)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=target_scenario[TARGET_PARAMS.BATCH_SIZE], shuffle=True)
    model_class = WESADNN if is_wesad else IHSNN

    central_model = model_class(
        data_scenario[DATA_PARAMS.IS_BINARY],
        all_train_predictors.shape[-1],
        target_scenario[TARGET_PARAMS.FIRST_LAYER_FEATURES],
        target_scenario[TARGET_PARAMS.LAYER_SCALINGS],
        target_scenario[TARGET_PARAMS.DROPOUT],
        use_batchnorm=True
    ).to(DEVICE)
    central_model.create_optimizer(target_scenario[TARGET_PARAMS.LEARNING_RATE])

    all_train_losses, all_train_accs = [], []
    for i in range(target_scenario[TARGET_PARAMS.EPOCHS]):
        central_model.train()
        for batch in train_loader:
            central_model.train_batch(batch)
            train_loss, train_acc, _ = central_model.compute_loss_acc(all_train_predictors, all_train_response, weights=all_train_weights)
            all_train_losses.append(train_loss)
            all_train_accs.append(train_acc)

        print("{} Train Loss: {} | Train {}: {}".format(i+1, train_loss, 'Acc' if data_scenario[DATA_PARAMS.IS_BINARY] else 'R2', train_acc))


    total_dict = {'results': {'loss': all_train_losses, 'accuracy': all_train_accs } }
    total_dict['scenarios'] = {
        'data': data_scenario, 
        'target': target_scenario
    }
    with open(results_path, 'wb') as file:
        pkl.dump(total_dict, file)


def run_no_protection_convergence_test(data_scenario, target_scenario, fl_scenario, defense_scenario, all_train_predictors, all_train_response, all_train_weights, client_data_dict, is_wesad=False):
    DEVICE = my_device()
    all_losses = []
    all_accs = []
    torch.manual_seed(fl_scenario[FL_PARAMS.TORCH_SEED])
    np.random.seed(fl_scenario[FL_PARAMS.NP_SEED])
    model_class = WESADNN if is_wesad else IHSNN

    fl_model = model_class(
        data_scenario[DATA_PARAMS.IS_BINARY],
        all_train_predictors.shape[-1],
        target_scenario[TARGET_PARAMS.FIRST_LAYER_FEATURES],
        target_scenario[TARGET_PARAMS.LAYER_SCALINGS],
        target_scenario[TARGET_PARAMS.DROPOUT]
    ).to(DEVICE)
    fl_model.create_optimizer(fl_scenario[FL_PARAMS.LEARNING_RATE])

    n_clients = len(client_data_dict.keys())
    clients_per_round = int(n_clients * fl_scenario[FL_PARAMS.CLIENT_PROP_PER_ROUND])

    scenario_losses = []
    scenario_accs = []
    print("CONVERGENCE NO PP")
    for i in range(fl_scenario[FL_PARAMS.GRADIENT_UPDATES]):
        client_ids = np.random.permutation(list(client_data_dict.keys()))[:clients_per_round]
        client_tuples = [client_data_dict[id] for id in client_ids]

        fl_model.train_batch_federated(
            client_dict_list=client_tuples, 
            defense_scenario=defense_scenario, 
            parameter_index=0, 
            no_protection_flag=True
        )

        train_loss, train_acc, _ = fl_model.compute_loss_acc(all_train_predictors, all_train_response, all_train_weights)
        print("{} Train Loss: {} | Train {}: {}".format(i, train_loss, "Acc" if fl_model.is_binary else "R2", train_acc))

        scenario_losses.append(train_loss)
        scenario_accs.append(train_acc)
    all_losses.append(scenario_losses)
    all_accs.append(scenario_accs)

    results_dict = {'defense_params': ["No PP"], 'loss':all_losses, 'accuracy':all_accs}
    total_dict = {'results':results_dict}
    return total_dict


def run_federated_convergence_tests(total_dict, data_scenario, target_scenario, fl_scenario, defense_scenario, all_train_predictors, all_train_response, all_train_weights, client_data_dict, results_path, is_wesad=False):
    DEVICE = my_device()
    all_losses = []
    all_accs = []

    for parameter_index in range(len(defense_scenario[DEFENSE_PARAMS.DEFENSE_STRATEGY_PARAMS][0])):
        print([def_list[parameter_index] for def_list in defense_scenario[DEFENSE_PARAMS.DEFENSE_STRATEGY_PARAMS]])
        
        torch.manual_seed(fl_scenario[FL_PARAMS.TORCH_SEED])
        np.random.seed(fl_scenario[FL_PARAMS.NP_SEED])
        model_class = WESADNN if is_wesad else IHSNN

        fl_model = model_class(
            data_scenario[DATA_PARAMS.IS_BINARY],
            all_train_predictors.shape[-1],
            target_scenario[TARGET_PARAMS.FIRST_LAYER_FEATURES],
            target_scenario[TARGET_PARAMS.LAYER_SCALINGS],
            target_scenario[TARGET_PARAMS.DROPOUT]
        ).to(DEVICE)
        fl_model.create_optimizer(fl_scenario[FL_PARAMS.LEARNING_RATE])

        n_clients = len(client_data_dict.keys())
        clients_per_round = int(n_clients * fl_scenario[FL_PARAMS.CLIENT_PROP_PER_ROUND])

        scenario_losses = []
        scenario_accs = []
        for i in range(fl_scenario[FL_PARAMS.GRADIENT_UPDATES]):
            client_ids = np.random.permutation(list(client_data_dict.keys()))[:clients_per_round]
            client_tuples = [client_data_dict[id] for id in client_ids] * fl_scenario[FL_PARAMS.DATA_MULTIPLIER]
            
            fl_model.train_batch_federated(
                client_dict_list=client_tuples, 
                defense_scenario=defense_scenario, 
                parameter_index=parameter_index, 
                no_protection_flag=False
            )
            train_loss, train_acc, _ = fl_model.compute_loss_acc(all_train_predictors, all_train_response, all_train_weights)
            print("{} Train Loss: {} | Train {}: {}".format(i, train_loss, "Acc" if fl_model.is_binary else "R2", train_acc))
            
            scenario_losses.append(train_loss)
            scenario_accs.append(train_acc)
        # END i in gradient updates

        all_losses.append(scenario_losses)
        all_accs.append(scenario_accs)
    # END parameter_index in
    total_dict['results']['defense_params'].extend(zip(*defense_scenario[DEFENSE_PARAMS.DEFENSE_STRATEGY_PARAMS]))
    total_dict['results']['loss'].extend(all_losses)
    total_dict['results']['accuracy'].extend(all_accs)
    total_dict['scenarios'] = {
        'data': data_scenario, 
        'target': target_scenario,
        'fl': fl_scenario,
        'defense': defense_scenario
    }
    with open(results_path, 'wb') as file:
        pkl.dump(total_dict, file)

def get_trained_attacker_model(known_yes_gradient_stack, known_no_gradient_stack, attack_scenario, scenario_index):
    DEVICE = my_device()
    attack_folds = attack_scenario[ATTACK_PARAMS.FOLDS]
    n_yes = known_yes_gradient_stack.shape[1]
    n_no = known_no_gradient_stack.shape[1]
    best_val_epoch = 0
    best_val_epochs = []

    torch.manual_seed(attack_scenario[ATTACK_PARAMS.TORCH_SEED])
    known_gradients = torch.cat((known_yes_gradient_stack, known_no_gradient_stack), dim=1).view((-1, known_yes_gradient_stack.shape[-1]))

    known_yes_label_stack = torch.ones(known_yes_gradient_stack.shape[0], known_yes_gradient_stack.shape[1], 1, dtype=torch.float, device=DEVICE)
    known_no_label_stack = torch.zeros(known_no_gradient_stack.shape[0], known_no_gradient_stack.shape[1], 1, dtype=torch.float, device=DEVICE)

    known_labels = torch.cat((known_yes_label_stack.view(-1, 1), known_no_label_stack.view(-1, 1)), dim=0)


    for k in range(attack_folds):
        
        shuffle_perm = torch.randperm(known_yes_gradient_stack.shape[1])
        train_yes_gradients = torch.cat((known_yes_gradient_stack[:, shuffle_perm, :][:, :int(k*n_yes/attack_folds)], known_yes_gradient_stack[:, shuffle_perm, :][:, int((k+1)*n_yes/attack_folds):]), dim=1).view((-1, known_yes_gradient_stack.shape[-1]))  
        train_yes_labels = torch.ones(train_yes_gradients.shape[0], 1, dtype=torch.float, device=DEVICE)
        val_yes_gradients = known_yes_gradient_stack[:, shuffle_perm, :][:, int(k*n_yes/attack_folds):int((k+1)*n_yes/attack_folds)].reshape((-1, known_yes_gradient_stack.shape[-1]))
        val_yes_labels = torch.ones(val_yes_gradients.shape[0], 1, dtype=torch.float, device=DEVICE)
        
        shuffle_perm = torch.randperm(known_no_gradient_stack.shape[1])    
        train_no_gradients = torch.cat((known_no_gradient_stack[:, shuffle_perm, :][:, :int(k*n_no/attack_folds)], known_no_gradient_stack[:, shuffle_perm, :][:, int((k+1)*n_no/attack_folds):]), dim=1).view((-1, known_no_gradient_stack.shape[-1]))  
        train_no_labels = torch.zeros(train_no_gradients.shape[0], 1, dtype=torch.float, device=DEVICE)
        val_no_gradients = known_no_gradient_stack[:, shuffle_perm, :][:, int(k*n_no/attack_folds):int((k+1)*n_no/attack_folds)].reshape((-1, known_no_gradient_stack.shape[-1]))
        val_no_labels = torch.zeros(val_no_gradients.shape[0], 1, dtype=torch.float, device=DEVICE)
        
        train_labels = torch.cat((train_yes_labels, train_no_labels), dim=0)
        val_labels = torch.cat((val_yes_labels, val_no_labels), dim=0)

        train_gradients = torch.cat((train_yes_gradients, train_no_gradients), dim=0)
        val_gradients = torch.cat((val_yes_gradients, val_no_gradients), dim=0)
    
        one_prop = train_labels.mean().item()
        train_weights = train_labels * (1 - one_prop) + (1-train_labels) * one_prop  
        train_ds = GradientDataset(
            train_gradients, 
            train_labels,
            train_weights
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size = attack_scenario[ATTACK_PARAMS.BATCH_SIZE],
            shuffle=True
        )
        torch.manual_seed(attack_scenario[ATTACK_PARAMS.TORCH_SEED] + k)
        grad_model = IHSNN(
            True,
            train_gradients.shape[-1],
            attack_scenario[ATTACK_PARAMS.FIRST_LAYER_FEATURES],
            attack_scenario[ATTACK_PARAMS.LAYER_SCALINGS],
            attack_scenario[ATTACK_PARAMS.DROPOUT],
            use_batchnorm=True
        ).to(DEVICE)
        grad_model.create_optimizer(attack_scenario[ATTACK_PARAMS.LEARNING_RATE], weight_decay=attack_scenario[ATTACK_PARAMS.WEIGHT_DECAY])

        best_val_acc = 0
        best_train_acc = 0
        print(train_labels.shape)
        print(val_labels.shape)
        for i in range(attack_scenario[ATTACK_PARAMS.MAX_TRAIN_EPOCHS]):

            for id, batch in enumerate(train_loader):
                grad_model.train_batch(batch)
                train_loss, train_acc, _ = grad_model.compute_loss_acc(train_gradients, train_labels, train_weights)
                val_loss, val_acc, _ = grad_model.compute_loss_acc(val_gradients, val_labels)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_train_acc = train_acc
                    best_val_epoch = i
            
        print("Best epoch: {} | Best val acc: {} | Corresponding train acc: {}".format(best_val_epoch, best_val_acc, best_train_acc))
        best_val_epochs.append(best_val_epoch)

    # Final Training
    one_prop = known_labels.mean().item()
    known_weights = known_labels * (1 - one_prop) + (1-known_labels) * one_prop  
    known_ds = GradientDataset(
        known_gradients, 
        known_labels,
        known_weights
    )
    known_loader = torch.utils.data.DataLoader(
        known_ds,
        batch_size = attack_scenario[ATTACK_PARAMS.BATCH_SIZE],
        shuffle=True
    )

    torch.manual_seed(attack_scenario[ATTACK_PARAMS.TORCH_SEED] + 99)
    grad_model = IHSNN(
        True,
        known_gradients.shape[-1],
        attack_scenario[ATTACK_PARAMS.FIRST_LAYER_FEATURES],
        attack_scenario[ATTACK_PARAMS.LAYER_SCALINGS],
        attack_scenario[ATTACK_PARAMS.DROPOUT],
        use_batchnorm=True
    ).to(DEVICE)
    grad_model.create_optimizer(attack_scenario[ATTACK_PARAMS.LEARNING_RATE], weight_decay=attack_scenario[ATTACK_PARAMS.WEIGHT_DECAY])
    
    best_train_acc = 0
    best_train_loss = 9999
    best_epoch = 0
    for i in range(sorted(best_val_epochs)[-1]):
        for id, batch in enumerate(known_loader):
            grad_model.train_batch(batch)
            train_loss, train_acc, _ = grad_model.compute_loss_acc(known_gradients, known_labels, known_weights)
            if train_loss < best_train_loss:
                best_train_acc = train_acc
                best_train_loss = train_loss
                best_epoch = i
                torch.save(grad_model.state_dict(), "temp_files/temp_attack_{}.sd".format(scenario_index))
                
        train_loss, train_acc, _ = grad_model.compute_loss_acc(known_gradients, known_labels, known_weights)
        print("{} Train Loss: {} | Train Acc: {}".format(i, train_loss, train_acc))
        
    print("Best Epoch {}".format(best_epoch))
    grad_model.load_state_dict(torch.load("temp_files/temp_attack_{}.sd".format(scenario_index)))
    return grad_model


def get_property_split(unscaled_df, np_seed, is_wesad=False):
    np.random.seed(np_seed)
    if is_wesad:
        labels_df = unscaled_df.groupby('client_id').agg({'Temp mean': 'mean'}) > unscaled_df['Temp mean'].mean() * 1.008
        high_mood_ids = labels_df.query('`Temp mean` == True').index.values
        low_mood_ids = labels_df.query('`Temp mean` == False').index.values
    else:
        labels_df = unscaled_df.groupby('STUDY_PRTCPT_ID').agg({'MOOD': 'mean'}) > unscaled_df['MOOD'].mean()
        high_mood_ids = labels_df.query('MOOD == True').index.values
        low_mood_ids = labels_df.query('MOOD == False').index.values
    return list(np.random.permutation(high_mood_ids)), list(np.random.permutation(low_mood_ids))

def run_no_protection_privacy_test(target_model, client_data_dict, unscaled_df, defense_scenario, attack_scenario, scenario_index, is_wesad=False):
    DEVICE = my_device()

    high_mood_ids, low_mood_ids = get_property_split(unscaled_df, attack_scenario[ATTACK_PARAMS.NP_SEED], is_wesad)
    num_high, num_low = len(high_mood_ids), len(low_mood_ids)
    attack_segs = attack_scenario[ATTACK_PARAMS.SEGMENTS]
    
    parameter_test_accs = []
    parameter_test_cm = []
    print("PRIVACY NO PP")
    for target_seg in range(attack_segs):
        segment_test_accs = []
        segment_test_cm = []
        print("SEGMENT {}/{}".format(target_seg+1, attack_segs))

        total_known_high_ids = high_mood_ids[:int(target_seg*num_high/attack_segs)] + high_mood_ids[int((target_seg+1)*num_high/attack_segs):]
        total_known_low_ids = low_mood_ids[:int(target_seg*num_low/attack_segs)] + low_mood_ids[int((target_seg+1)*num_low/attack_segs):]
        
        for t in range(attack_scenario[ATTACK_PARAMS.TEST_REPEATS]):
            print("TEST SLICE {}/{}".format(t+1, attack_scenario[ATTACK_PARAMS.TEST_REPEATS]))
            known_high_ids = total_known_high_ids[int(len(total_known_high_ids) * attack_scenario[ATTACK_PARAMS.DATA_ACCESS_MULTI])*t:int(len(total_known_high_ids) * attack_scenario[ATTACK_PARAMS.DATA_ACCESS_MULTI])*(t+1)]
            known_low_ids = total_known_low_ids[int(len(total_known_low_ids) * attack_scenario[ATTACK_PARAMS.DATA_ACCESS_MULTI])*t:int(len(total_known_low_ids) * attack_scenario[ATTACK_PARAMS.DATA_ACCESS_MULTI])*(t+1)]
        
            print("Total Clients: " + str(num_high + num_low))
            print("Compromised Data Amount: {} high + {} low".format(len(known_high_ids), len(known_low_ids)))

            torch.manual_seed(attack_scenario[ATTACK_PARAMS.TORCH_SEED] + 4242 + target_seg)
            known_yes_gradient_stack = target_model.get_gradient_stack(
                [client_data_dict[client_id] for client_id in known_high_ids], 
                repeats=attack_scenario[ATTACK_PARAMS.TRAIN_SAMPLE_MULTI]
            )

            known_no_gradient_stack = target_model.get_gradient_stack(
                [client_data_dict[client_id] for client_id in known_low_ids], 
                repeats=attack_scenario[ATTACK_PARAMS.TRAIN_SAMPLE_MULTI]
            )

            if attack_scenario[ATTACK_PARAMS.NORMALIZE_GRADIENTS]:
                known_yes_gradient_stack = known_yes_gradient_stack/torch.unsqueeze(torch.linalg.norm(known_yes_gradient_stack, ord=2, dim=-1), dim=-1)
                known_no_gradient_stack = known_no_gradient_stack/torch.unsqueeze(torch.linalg.norm(known_no_gradient_stack, ord=2, dim=-1), dim=-1)
            elif defense_scenario[DEFENSE_PARAMS.SCALE_GRADIENT_BY_SAMPLE_SIZE]:
                yes_sample_sizes = torch.tensor([[client_data_dict[client_id]['labels'].shape[0]] for client_id in known_high_ids]).to(DEVICE)
                known_yes_gradient_stack *= yes_sample_sizes
                no_sample_sizes = torch.tensor([[client_data_dict[client_id]['labels'].shape[0]] for client_id in known_low_ids]).to(DEVICE)
                known_yes_gradient_stack *= no_sample_sizes

            attacker_model = get_trained_attacker_model(known_yes_gradient_stack, known_no_gradient_stack, attack_scenario, scenario_index)
            attacker_model.eval()

            del known_yes_gradient_stack
            del known_no_gradient_stack

            test_high_ids = high_mood_ids[int(target_seg*num_high/attack_segs):int((target_seg+1)*num_high/attack_segs)]
            test_low_ids = low_mood_ids[int(target_seg*num_low/attack_segs):int((target_seg+1)*num_low/attack_segs)]
            print(test_high_ids)
            print(test_low_ids)

            test_labels = torch.tensor(([1] * len(test_high_ids) + [0] * len(test_low_ids)) * attack_scenario[ATTACK_PARAMS.TEST_SAMPLE_MULTI], device=DEVICE, dtype=torch.float).view((-1, 1))

            torch.manual_seed(attack_scenario[ATTACK_PARAMS.TORCH_SEED] + 4242 + target_seg)
            test_gradient_stack = target_model.get_gradient_stack(
                [client_data_dict[client_id] for client_id in (test_high_ids + test_low_ids)],
                repeats=attack_scenario[ATTACK_PARAMS.TEST_SAMPLE_MULTI]
            )

            if attack_scenario[ATTACK_PARAMS.NORMALIZE_GRADIENTS]:
                test_gradient_stack = test_gradient_stack/torch.unsqueeze(torch.linalg.norm(test_gradient_stack, ord=2, dim=-1), dim=-1)
            elif defense_scenario[DEFENSE_PARAMS.SCALE_GRADIENT_BY_SAMPLE_SIZE]:
                sample_sizes = torch.tensor([[client_data_dict[client_id]['labels'].shape[0]] for client_id in test_high_ids + test_low_ids]).to(DEVICE)
                test_gradient_stack *= sample_sizes

            test_gradients = test_gradient_stack.view((-1, test_gradient_stack.shape[-1]))    
            test_loss, test_acc, test_cm = attacker_model.compute_loss_acc(test_gradients, test_labels)
            print(test_acc)
            print(test_cm)
            segment_test_accs.append(test_acc)
            segment_test_cm.append(test_cm)

        parameter_test_accs.append(segment_test_accs)
        parameter_test_cm.append(segment_test_cm)

    final_results = {
        'conditions': ['No PP'],
        'measures': [parameter_test_accs], 
        'cm': [parameter_test_cm]
    }
    final_dict = {'results': final_results}
    
    return final_dict
  

def run_privacy_tests(final_dict, data_scenario, target_scenario, defense_scenario, attack_scenario, target_model, unscaled_df, client_data_dict, results_filename, scenario_index, is_wesad=False):
    DEVICE = my_device()
    
    high_mood_ids, low_mood_ids = get_property_split(unscaled_df, attack_scenario[ATTACK_PARAMS.NP_SEED], is_wesad)
    num_high, num_low = len(high_mood_ids), len(low_mood_ids)

    attack_segs = attack_scenario[ATTACK_PARAMS.SEGMENTS]
    attack_folds = attack_scenario[ATTACK_PARAMS.FOLDS]

    scale_by_samples = defense_scenario[DEFENSE_PARAMS.SCALE_GRADIENT_BY_SAMPLE_SIZE]
    normalize = attack_scenario[ATTACK_PARAMS.NORMALIZE_GRADIENTS]

    all_test_accs = [[[] for _a in range(attack_segs)] for _b in range(len(defense_scenario[DEFENSE_PARAMS.DEFENSE_STRATEGY_PARAMS][0]))]
    all_test_cm = [[[] for _a in range(attack_segs)] for _b in range(len(defense_scenario[DEFENSE_PARAMS.DEFENSE_STRATEGY_PARAMS][0]))]
    
    # Do tests
    print("PRIVACY TESTS")
    for target_seg in range(attack_segs):
        print("SEGMENT {}/{}".format(target_seg+1, attack_scenario[ATTACK_PARAMS.SEGMENTS]))

        total_known_high_ids = high_mood_ids[:int(target_seg*num_high/attack_segs)] + high_mood_ids[int((target_seg+1)*num_high/attack_segs):]
        total_known_low_ids = low_mood_ids[:int(target_seg*num_low/attack_segs)] + low_mood_ids[int((target_seg+1)*num_low/attack_segs):]
        
        for t in range(attack_scenario[ATTACK_PARAMS.TEST_REPEATS]):
            print("TEST SLICE {}/{}".format(t+1, attack_scenario[ATTACK_PARAMS.TEST_REPEATS]))

            known_high_ids = total_known_high_ids[int(len(total_known_high_ids) * attack_scenario[ATTACK_PARAMS.DATA_ACCESS_MULTI])*t:int(len(total_known_high_ids) * attack_scenario[ATTACK_PARAMS.DATA_ACCESS_MULTI])*(t+1)]
            known_low_ids = total_known_low_ids[int(len(total_known_low_ids) * attack_scenario[ATTACK_PARAMS.DATA_ACCESS_MULTI])*t:int(len(total_known_low_ids) * attack_scenario[ATTACK_PARAMS.DATA_ACCESS_MULTI])*(t+1)]
            print("Total Clients: " + str(num_high + num_low))
            print("Compromised Data Amount: {} high + {} low".format(len(known_high_ids), len(known_low_ids)))

            torch.manual_seed(attack_scenario[ATTACK_PARAMS.TORCH_SEED] + 4242 + target_seg)
            base_known_yes_gradient_stack = target_model.get_gradient_stack(
                [client_data_dict[client_id] for client_id in known_high_ids], 
                repeats=attack_scenario[ATTACK_PARAMS.TRAIN_SAMPLE_MULTI]
            ).cpu()
            base_known_no_gradient_stack = target_model.get_gradient_stack(
                [client_data_dict[client_id] for client_id in known_low_ids], 
                repeats=attack_scenario[ATTACK_PARAMS.TRAIN_SAMPLE_MULTI]
            ).cpu()


            test_high_ids = high_mood_ids[int(target_seg*num_high/attack_segs):int((target_seg+1)*num_high/attack_segs)]
            test_low_ids = low_mood_ids[int(target_seg*num_low/attack_segs):int((target_seg+1)*num_low/attack_segs)]
            test_labels = torch.tensor(([1] * len(test_high_ids) + [0] * len(test_low_ids)) * attack_scenario[ATTACK_PARAMS.TEST_SAMPLE_MULTI], device=DEVICE, dtype=torch.float).view((-1, 1))

        
            torch.manual_seed(attack_scenario[ATTACK_PARAMS.TORCH_SEED] + 4242 + target_seg)
            base_test_gradient_stack = target_model.get_gradient_stack(
                [client_data_dict[client_id] for client_id in (test_high_ids + test_low_ids)],
                attack_scenario[ATTACK_PARAMS.TEST_SAMPLE_MULTI]
            ).cpu()
        
            get_zero_noise_model = True
            for parameter_index in range(len(defense_scenario[DEFENSE_PARAMS.DEFENSE_STRATEGY_PARAMS][0])):
                print([s[parameter_index] for s in defense_scenario[DEFENSE_PARAMS.DEFENSE_STRATEGY_PARAMS]])
                if attack_scenario[ATTACK_PARAMS.NOISE_IS_KNOWN]:

                    known_yes_gradient_stack = modify_gradients(
                        base_known_yes_gradient_stack,
                        defense_scenario,
                        parameter_index,
                        attack_scenario[ATTACK_PARAMS.CLIP_IS_KNOWN],
                        attack_scenario[ATTACK_PARAMS.NORMALIZE_GRADIENTS],
                        for_training=False         
                    ).to(DEVICE)

                    known_no_gradient_stack = modify_gradients(
                        base_known_no_gradient_stack,
                        defense_scenario,
                        parameter_index,
                        attack_scenario[ATTACK_PARAMS.CLIP_IS_KNOWN],
                        attack_scenario[ATTACK_PARAMS.NORMALIZE_GRADIENTS],
                        for_training=False         
                    ).to(DEVICE)


                    if defense_scenario[DEFENSE_PARAMS.SCALE_GRADIENT_BY_SAMPLE_SIZE]:
                        # sample_sizes = torch.tensor([[client_data_dict[client_id]['labels'].shape[0]] for client_id in known_ids] * attack_scenario[ATTACK_PARAMS.TRAIN_SAMPLE_MULTI]).to(DEVICE)
                        # known_gradient_stack *= sample_sizes.view((known_gradient_stack.shape[0], known_gradient_stack.shape[1], -1))
                        # del sample_sizes
                        pass

                    attacker_model = get_trained_attacker_model(
                        known_yes_gradient_stack,
                        known_no_gradient_stack,
                        attack_scenario,
                        scenario_index
                    )
                    attacker_model.eval()

                    if DEFENSE_STRATEGY.GRADIENT_CENSOR in defense_scenario[DEFENSE_PARAMS.DEFENSE_STRATEGY_LIST]:
                        # zero_vector = known_gradient_stack.view((-1, known_gradient_stack.shape[-1]))[0] == 0
                        pass
                     # del known_gradient_stack
                else:
                    # if get_zero_noise_model:
                    #     print("Attacker does not know noise, generating zero noise model")
                    #     known_gradient_stack = modify_gradients(
                    #         base_known_gradient_stack,
                    #         defense_scenario,
                    #         0,
                    #         attack_scenario[ATTACK_PARAMS.CLIP_IS_KNOWN],
                    #         attack_scenario[ATTACK_PARAMS.NORMALIZE_GRADIENTS],
                    #         for_training=False         
                    #     ).to(DEVICE)

                    # if defense_scenario[DEFENSE_PARAMS.SCALE_GRADIENT_BY_SAMPLE_SIZE]:
                    #     sample_sizes = torch.tensor([[client_data_dict[client_id]['labels'].shape[0]] for client_id in known_ids] * attack_scenario[ATTACK_PARAMS.TRAIN_SAMPLE_MULTI]).to(DEVICE)
                    #     known_gradient_stack *= sample_sizes.view((known_gradient_stack.shape[0], known_gradient_stack.shape[1], -1))
                    #     del sample_sizes

                    # attacker_model = get_trained_attacker_model(
                    #     known_gradient_stack,
                    #     known_label_stack,
                    #     attack_scenario,
                    #     scenario_index
                    # )
                    # attacker_model.eval()
                    # del known_gradient_stack
                    # del known_label_stack
                    # get_zero_noise_model = False
                    pass 
                # END IF noise is known

                test_gradient_stack = modify_gradients(
                    base_test_gradient_stack,
                    defense_scenario,
                    parameter_index,
                    clip_is_known=True,
                    normalize=attack_scenario[ATTACK_PARAMS.NORMALIZE_GRADIENTS],
                    for_training=False     
                ).to(DEVICE)

                if defense_scenario[DEFENSE_PARAMS.SCALE_GRADIENT_BY_SAMPLE_SIZE]:
                    # sample_sizes = torch.tensor([[client_data_dict[client_id]['labels'].shape[0]] for client_id in test_ids] * attack_scenario[ATTACK_PARAMS.TEST_SAMPLE_MULTI]).to(DEVICE)
                    # test_gradient_stack *= sample_sizes.view((test_gradient_stack.shape[0], test_gradient_stack.shape[1], -1))
                    pass 
                
                test_gradients = test_gradient_stack.view((-1, test_gradient_stack.shape[-1]))
        
                test_loss, test_acc, test_cm = attacker_model.compute_loss_acc(test_gradients, test_labels)
                print(test_acc)
                print(test_cm)
                all_test_accs[parameter_index][target_seg].append(test_acc)
                all_test_cm[parameter_index][target_seg].append(test_cm)
                del test_gradients
                del test_gradient_stack
        # END for test_repeats
        # END for parameter_index
    # END for target_seg
        

    final_dict['results']['conditions'].extend(zip(*defense_scenario[DEFENSE_PARAMS.DEFENSE_STRATEGY_PARAMS]))
    final_dict['results']['measures'].extend(all_test_accs)
    final_dict['results']['cm'].extend(all_test_cm)
    final_dict['scenarios'] = {
        'data': data_scenario, 
        'target': target_scenario, 
        'defense': defense_scenario, 
        'attack': attack_scenario
    }
    
    with open(results_filename, 'wb') as file:
        pkl.dump(final_dict, file)