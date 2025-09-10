from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import lfp.lfp_analysis.plotting as lfplt
import lfp.lfp_analysis.event_extraction as ee
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, product
import multiprocessing
import pickle
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import math 
from scipy.stats import sem

class CustomCallback(xgb.callback.TrainingCallback):
    def __init__(self):
        super().__init__()
        self.best_score = float('inf')
        self.best_iteration = None
        self.current_round = 0

    def after_iteration(self, model, epoch, evals_log):
        current_score = evals_log["test"]["auc"][-1][0]
        self.current_round += 1
        if current_score < self.best_score:
            self.best_score = current_score
            self.best_iteration = self.current_round
            self.best_model = model

# myCallback = CustomCallback()
#         results = xgb.cv(params, dtrain, num_boost_round = 10, stratified = True, nfold=num_fold, metrics={"auc"}, seed=0, fpreproc=fpreproc , callbacks=[myCallback] )
#         shuffle_results = xgb.cv(
#             params, dtrain, num_boost_round=5, stratified=True, metrics=["auc"], nfold=num_fold, seed=0)  

def trial_decoder(lfp_collection, num_fold, mode, events, baseline=None, event_len=None, pre_window=0, post_window=0):
    # for power: data = [trials,  frequencies, brain region,]
    # for not power: data = [trials, freqeuncies, brain region, brain region]
    # decoder data = [trials, ...] for each band
    data_dict = {event: None for event in events}
    features = [] 
    if 'power' in mode:
        data_dict, features = __prep_feature_data__(lfp_collection, events, baseline, event_len, pre_window, post_window, 'power', data_dict, features)
    if 'coherence' in mode:
        data_dict, features = __prep_feature_data__(lfp_collection, events, baseline, event_len, pre_window, post_window,'coherence', data_dict, features)
    if 'granger' in mode: 
        data_dict, features = __prep_feature_data__(lfp_collection,events, baseline, event_len, pre_window, post_window, 'granger', data_dict, features)
    results_dict = {'features': features}
    for event in events:
        results_dict[event] = {'auc': [], 'prob': [], 'weights': [], 'models':[], 'auc shuffle': []}
        data, labels = __prep_data__(data_dict, events, event)
        splits = {}
        skf = StratifiedKFold(n_splits = num_fold, shuffle = True)
        for i, (train_index, test_index) in enumerate(skf.split(data, labels)):
            splits[i] = [train_index, test_index]
        for fold in range(num_fold):
            train_data, test_data, train_labels, test_labels = __prep_train_test_feature_data__(fold, splits, data, labels)
            pred_rf, auc_rf, feat_imp_rf, model_rf  = __run_model__(train_data, train_labels, test_data, test_labels)
            results_dict[event]['auc'].append(auc_rf)
            results_dict[event]['prob'].append(pred_rf)
            results_dict[event]['weights'].append(feat_imp_rf)
            results_dict[event]['models'].append(model_rf)
            pred_shuf, auc_shuf, feat_imp_shuf, model_shuf  = __run_model__(train_data, train_labels, test_data, np.random.permutation(test_labels))
            results_dict[event]['auc shuffle'].append(auc_shuf)
    results_object =  all_results(results_dict, num_fold, event_len, pre_window, post_window)
    return results_object

def __reshape_data__(lfp_collection, agent_band_dict, mode):
    decoder_data = {}
    for event, bands in agent_band_dict.items():
        stacked_bands = np.stack(list(bands.values()), axis=0)
        band_names = list(bands.keys())
        # stacked bands = [band, trials, regions]
        # what i want = [trials, bandxregions]
        if mode != "coherence":
            if mode == "power":
                reshaped_bands = np.transpose(stacked_bands, (1, 0, 2))
                features = list(product(range(stacked_bands.shape[0]), range(stacked_bands.shape[2])))
                reshaped_bands = reshaped_bands.reshape(reshaped_bands.shape[0], -1)
        if mode == "coherence":
            reshaped_bands, features = __reshape_coherence_data__(stacked_bands)
        if mode == "granger":
            reshaped_bands, features = __reshape_granger_data__(stacked_bands)
        decoder_data[event] = reshaped_bands
        feature_names = get_feature_names(features, lfp_collection.brain_region_dict, band_names, mode)
    return decoder_data, features, feature_names

def __reshape_coherence_data__(stacked_bands):
    n_bands, n_trials, n_regions, _ = stacked_bands.shape
    # Get indices for upper triangle (excluding diagonal)
    region_pairs = list(combinations(range(n_regions), 2))
    # Initialize output array
    # Shape will be [trials, bands * number_of_unique_pairs]
    n_pairs = len(region_pairs)
    reshaped = np.zeros((n_trials, n_bands * n_pairs))
    feature_indices = []
    # Fill the array
    for band in range(n_bands):
        for pair_idx, (i, j) in enumerate(region_pairs):
            # Get position in final array
            output_idx = band * n_pairs + pair_idx
            reshaped[:, output_idx] = stacked_bands[band, :, i, j]
            feature_indices.append(tuple([band, i, j]))
    return reshaped, feature_indices


def __reshape_granger_data__(stacked_bands):
    n_bands, n_trials, n_regions, _ = stacked_bands.shape
    # Get off-diagonal indices
    region_pairs = [(i, j) for i, j in product(range(n_regions), range(n_regions)) if i != j]
    # Initialize output array
    n_pairs = len(region_pairs)
    reshaped = np.zeros((n_trials, n_bands * n_pairs))
    feature_indices = []
    # Fill the array
    for band in range(n_bands):
        for pair_idx, (i, j) in enumerate(region_pairs):
            output_idx = band * n_pairs + pair_idx
            reshaped[:, output_idx] = stacked_bands[band, :, i, j]
            feature_indices.append(tuple([band, i, j]))
    return reshaped, feature_indices


def __prep_data__(decoder_data, events, event):
    data_neg = []
    data_pos = []
    for trial in decoder_data[event]:
        data_pos.append(trial)
    for neg_event in np.setdiff1d(events, event):
        for trial in decoder_data[neg_event]:
            data_neg.append(trial)
    data_pos = np.stack(data_pos)
    data_neg = np.stack(data_neg)
    label_pos = np.ones(data_pos.shape[0])
    label_neg = np.zeros(data_neg.shape[0])
    data = np.concatenate([data_pos, data_neg], axis=0)
    # data = (samples, features, timebins)
    labels = np.concatenate([label_pos, label_neg], axis=0)
    shuffle = np.random.permutation(len(labels))
    data = data[shuffle, :]
    labels = labels[shuffle]
    return data, labels


def get_feature_names(features, brain_region_dict, band_names, mode):
    feature_names = []
    if mode == "power":
        for band_idx, region_idx in features:
            name = f"{band_names[band_idx]}_{brain_region_dict.inverse[region_idx]}"
            feature_names.append(name)
    else:  # coherence or granger
        for band_idx, reg1_idx, reg2_idx in features:
            if mode =='coherence':
                name = f"{band_names[band_idx]}_{brain_region_dict.inverse[reg1_idx]}_{brain_region_dict.inverse[reg2_idx]}"
            else:
                name = f"{band_names[band_idx]}_from{brain_region_dict.inverse[reg1_idx]}_to{brain_region_dict.inverse[reg2_idx]}"
            feature_names.append(name)
    return feature_names

def calc_top_feat_trial_decoder(lfp_collection, num_features,
    num_fold, events, mode, top_features=None, baseline=None, event_len=None, pre_window=0, post_window=0
):
    data_dict = {event: None for event in events}
    features = [] 
    if 'power' in mode:
        data_dict, features = __prep_feature_data__(lfp_collection,events, baseline, event_len, pre_window, post_window, 'power', data_dict, features)
    if 'coherence' in mode:
        data_dict, features = __prep_feature_data__(lfp_collection,events, baseline, event_len, pre_window, post_window,'coherence', data_dict, features)
    if 'granger' in mode: 
        data_dict, features = __prep_feature_data__(lfp_collection,events, baseline, event_len, pre_window, post_window, 'granger', data_dict, features)
    feature_dict = {}
    for event in events:
        feature_indices = []
        data, labels = __prep_data__(data_dict, events, event)
        splits = {}
        skf = StratifiedKFold(n_splits = num_fold, shuffle = True)
        for i, (train_index, test_index) in enumerate(skf.split(data, labels)):
            splits[i] = [train_index, test_index]
        for feat in range(num_features): 
            feats_tested = []
            all_auc_rf = []
            for i in range(len(features)):
                temp_indices = feature_indices.copy()
                feat_auc_rf = []
                if i not in feature_indices: 
                    temp_indices.append(i)
                    for fold in range(num_fold):
                        train_data, test_data, train_labels, test_labels = __prep_train_test_feature_data__(fold, splits, data, labels, temp_indices)
                        pred_rf, auc_rf, feat_imp_rf, model_rf  = __run_model__(train_data, train_labels, test_data, test_labels)
                        feat_auc_rf.append(auc_rf)
                    feat_auc_rf = np.nanmean(np.array(feat_auc_rf))
                    all_auc_rf.append(feat_auc_rf)
                    feats_tested.append(i)
            max_index = np.argmax(np.array(all_auc_rf))
            feature_indices.append(feats_tested[max_index])
        top_features = [features[i] for i in feature_indices]
        feature_dict[event] = [feature_indices, top_features]
     
    return feature_dict
        
def __prep_feature_data__(lfp_collection, events, baseline, event_len, pre_window, post_window, mode, data_dict, all_features):
    data = ee.average_events(lfp_collection,
                events=events,
                mode=mode,
                baseline=baseline,
                event_len=event_len,
                pre_window=pre_window,
                post_window=post_window,
                plot=False,
            )
    [agent_band_dict, band_agent_dict] = lfplt.band_calcs(data)
    data, features, feature_names = __reshape_data__(lfp_collection, agent_band_dict, mode)
    band_names = agent_band_dict.keys()
    for event in data_dict.keys():
        if data_dict[event] is None:
            temp = np.empty((data[event].shape[0],0))
            new = np.concatenate([temp, data[event]], axis=1)
            data_dict[event] = new
        else:                  
            data_dict[event] = np.concatenate([data_dict[event], data[event]], axis = 1)
    all_features.extend(feature_names)
    return data_dict, all_features

def __prep_train_test_feature_data__(fold, splits, data, labels, feature_indices=None):
    train_data = data[splits[fold][0], :]
    train_labels = labels[splits[fold][0]]
    test_data = data[splits[fold][1], :]
    test_labels = labels[splits[fold][1]]
    if feature_indices is not None:
        train_data = train_data[:, feature_indices]
        test_data = test_data[:, feature_indices]
        if len(train_data.shape) == 1:
            train_data = train_data.reshape(-1,1)
            test_data = test_data.reshape(-1,1)
    return train_data, test_data, train_labels, test_labels


def __run_model__(train_data, train_labels, test_data, test_labels):
    model_rf = xgb.XGBClassifier(n_estimators=100, max_depth=6, objective='binary:logistic', n_jobs = multiprocessing.cpu_count())
    model_rf.fit(X=train_data, y=train_labels, sample_weight=compute_sample_weight("balanced", train_labels))
    pred_rf = model_rf.predict_proba(test_data)
    feat_imp_rf = model_rf.feature_importances_
    auc_rf = (roc_auc_score(test_labels, pred_rf[:, 1]))
    return pred_rf, auc_rf, feat_imp_rf, model_rf
                                                                                       
                                                    
# TODO: UNFINISHED OR UNUSED

def __probabilities__(results, labels, t_data, num_fold):
    probabilities = []
    prob_labels = []
    for i in range(num_fold):
        test_indices = results["indices"]["test"][i]
        test_data = t_data[test_indices, :]
        test_labels = labels[test_indices]
        model = results["estimator"][i]
        prob = model.predict_proba(test_data)
        probabilities.append(prob)
        prob_labels.append(test_labels)
    prob_dict = {"probabilities": probabilities, "labels": prob_labels}
    return prob_dict


class all_results:
    def __init__(self, results_dict, num_fold, event_length, pre_window, post_window):
        self.num_fold = num_fold
        self.events = list(results_dict.keys())
        self.events.remove('features')
        self.event_length = event_length
        self.pre_window = pre_window
        self.post_window = post_window
        self.features = results_dict['features']
        results = {}
        for event in self.events:
            results[event] = model_results(results_dict[event], self.features)
        self.results = results

    def __repr__(self):
        output = [f"Models ran with {self.num_fold} folds"]
        output.append(f"Events: {self.events}")
        for label, results in self.results.items():
            output.append(f"  {label}: {repr(results)}")
        return "\n".join(output)


    def plot_average(self, start=0, stop=None):
        no_plots = len(self.events)
        height_fig = math.ceil(no_plots / 2)
        i = 1
        bar_width = 0.2
        #total_event = self.event_length + self.post_window
        plt.figure(figsize=(8, 4 * height_fig))
        for key, results in self.results.items():
            plt.subplot(height_fig, 2, i)
            # x = np.linspace(-self.pre_window, total_event, np.array(results.roc_auc).shape[0])
            # if start is not None:
            #     start = np.where(x >= start)[0][0]
            # if stop is None:
            #     stop = len(results.auc)
            # # if stop is not None:
            #     stop = np.where(x <= stop)[0][-1] + 1
            rf_avg = np.mean(np.mean(results.auc))
            rf_sem = sem(np.mean(results.auc))
            rf_shuffle_avg = np.mean(np.mean(results.shuffle))
            rf_shuffle_sem = sem(np.mean(results.shuffle))
            bar_positions = np.array([0.3, 0.6])
            plt.bar(bar_positions[0], rf_avg, bar_width, label="RF", yerr=rf_sem, capsize=5)
            plt.bar(bar_positions[1], rf_shuffle_avg, bar_width, label="RF Shuffle", yerr=rf_shuffle_sem, capsize=5)
            plt.title(f"{key}")
            plt.ylim(0, 1)
            if i == 2:
                plt.legend(bbox_to_anchor=(1, 1))
            i += 1
            plt.xticks([])
        plt.suptitle("Decoder Accuracy")
        plt.show()

class model_results:
    def __init__(self, model_dict, features):
        self.models = model_dict['models']
        self.shuffle = model_dict['auc shuffle']
        self.auc = model_dict['auc']
        self.features = features
        self.num_fold = len(model_dict['weights'])
        probabilites = []
        for i in range(self.num_fold):
            probabilites.append(model_dict['prob'][i])
        self.avg_auc = np.mean(np.array(self.auc))
        self.avg_shuffle = np.mean(np.array(self.shuffle))
        self.__config_weights__(model_dict)
        
    def __config_weights__(self, model_dict):
        fold_arrays = model_dict['weights']
        data = {'Feature': self.features}
        # Add fold columns
        for i, fold_array in enumerate(fold_arrays):
            data[f'Fold {i+1}'] = fold_array
        # Create DataFrame and add average column
        df = pd.DataFrame(data)
        df['Average Weight'] = df.filter(like='Fold').mean(axis=1)
        df.sort_values('Average Weight', ascending = False, inplace = True)
        self.feature_df = df 

    def __repr__(self):
        output = ["Model Results"]
        output.append(f"Average AUC score: {self.avg_auc}")
        output.append(f"Average AUC score for shuffled data: {self.avg_shuffle}")
        # output.append(f"Total positive trials:{self.pos_labels}: Total neg trials:{self.neg_labels}")
        return "\n".join(output)

    def get_feature_imp(self):
        return self.feature_df

def get_feature_indices(top_features):
    band_dict = {"delta": 0, "theta": 1, "beta": 2, "low_gamma": 3, "high_gamma": 4}
    top_power_indices = []
    top_coherence_indices = []
    for feature in np.unique(top_features):
        brain_region = feature.split(" ")[0]
        band = feature.split(" ")[1:]
        if len(band) == 2:
            band = band[0] + "_" + band[1]
        else:
            band = band[0]
        band_index = band_dict[band]
        try:
            brain_index = test_analysis.brain_region_dict[brain_region]
            power_index = band_index * 5 + brain_index
            top_power_indices.append(power_index)
        except KeyError:
            brain_index = test_analysis.coherence_pairs_dict[brain_region]
            coherence_index = band_index * 10 + brain_index
            top_coherence_indices.append(coherence_index)
    return (sorted(top_power_indices), sorted(top_coherence_indices))