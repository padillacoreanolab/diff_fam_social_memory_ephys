from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import lfp.lfp_analysis.LFP_analysis as lfp
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, product
import multiprocessing


def trial_decoder(lfp_collection, num_fold, mode, events, baseline=None, event_len=None, pre_window=0, post_window=0):
    data = lfp.average_events(
        lfp_collection,
        events=events,
        mode=mode,
        baseline=baseline,
        event_len=event_len,
        pre_window=pre_window,
        post_window=post_window,
        plot=False,
    )
    # for power: data = [trials,  frequencies, brain region,]
    # for not power: data = [trials, freqeuncies, brain region, brain region]
    [agent_band_dict, band_agent_dict] = lfp.band_calcs(data)
    # decoder data = [trials, ...] for each band
    results_dict = {}
    shuffle_results_dict = {}
    event_labels = {}
    decoder_data = {}
    params = {
        "max_depth": 6,
        "objective": "binary:logistic",
        "n_jobs": multiprocessing.cpu_count(),
    }
    decoder_data, features = __reshape_data__(agent_band_dict, mode)
    for event in events:
        data, labels = __prep_data__(decoder_data, events, event)
        dtrain = xgb.DMatrix(data, labels)
        dshuffle = xgb.DMatrix(data, np.random.permutation(labels))
        # data = [samples, features]
        results = xgb.cv(
            params,
            dtrain,
            num_boost_round=5,
            stratified=True,
            metrics=["auc", "fmap"],
            nfold=num_fold,
            seed=0,
        )
        shuffle_results = xgb.cv(
            params, dshuffle, num_boost_round=5, stratified=True, metrics=["auc"], nfold=num_fold, seed=0
        )

    #     prob_dict = __probabilities__(results, labels, data, num_fold)
    #     results["probabilities"] = prob_dict
    #     results_dict[event].append(results)
    #     shuffle_results_dict[event].append(shuffle_results)
    # result_object = all_results(results_dict, shuffle_results_dict, num_fold, event_labels, pre_window, post_window)
    return results, shuffle_results


def __reshape_data__(lfp_collection, agent_band_dict, mode):
    decoder_data = {}
    for event, bands in agent_band_dict.items():
        stacked_bands = np.stack(list(bands.values()), axis=0)
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
        feature_names = get_feature_names(features, lfp_collection.brain_region_dict, mode)
    return decoder_data, features


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


def get_feature_names(features, brain_region_dict, mode):
    feature_names = []
    if mode == "power":
        for band_idx, region_idx in features:
            name = f"{band_names[band_idx]}_{brain_region_dict.inverse[region_idx]}"
            feature_names.append(name)
    else:  # coherence or granger
        for band_idx, reg1_idx, reg2_idx in features:
            name = f"{band_names[band_idx]}_{brain_region_dict[reg1_idx]}_{brain_region_dict.inverse[reg2_idx]}"
            feature_names.append(name)

    return feature_names


def __train_test_split__(fold, num_fold, data_pos, data_neg, num_pos, num_neg):
    pos_fold = num_pos // num_fold
    neg_fold = num_neg // num_fold
    data_test = np.concatenate(
        (data_pos[fold * pos_fold : (fold + 1) * pos_fold, :], data_neg[fold * neg_fold : (fold + 1) * neg_fold, :]),
        axis=0,
    )
    label_test = np.concatenate(
        (np.ones((fold + 1) * pos_fold - fold * pos_fold), np.zeros((fold + 1) * neg_fold - fold * neg_fold))
    )
    data_train = np.concatenate(
        (
            data_pos[np.setdiff1d(np.arange(num_pos), np.arange(fold * pos_fold, (fold + 1) * pos_fold)), :],
            data_neg[np.setdiff1d(np.arange(num_neg), np.arange(fold * neg_fold, (fold + 1) * neg_fold)), :],
        ),
        axis=0,
    )
    label_train = np.concatenate(
        (
            np.ones(num_pos - (fold + 1) * pos_fold + fold * pos_fold),
            np.zeros(num_neg - (fold + 1) * neg_fold + fold * neg_fold),
        )
    )
    return (data_test, label_test, data_train, label_train)


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
    def __init__(self, results_dict, shuffle_dict, num_fold, event_labels, event_length, pre_window, post_window):
        self.num_fold = num_fold
        self.events = results_dict.keys()
        self.event_length = event_length
        self.pre_window = pre_window
        self.post_window = post_window
        results = {}
        for event in self.events:
            results[event] = model_results(results_dict[event], shuffle_dict[event], event_labels[event], num_fold)
        self.results = results

    def __repr__(self):
        output = [f"Models ran with {self.num_fold} folds"]
        output.append(f"Events: {self.events}")
        for label, results in self.results.items():
            output.append(f"  {label}: {repr(results)}")
        return "\n".join(output)

    def plot_across_time(self, start=None, stop=None):
        no_plots = len(self.events)
        height_fig = math.ceil(no_plots / 2)
        i = 1
        if start is None:
            start = -self.pre_window
        if stop is None:
            stop = self.event_length + self.post_window
        plt.figure(figsize=(12, 4 * height_fig))
        for key, results in self.results.items():
            plt.subplot(height_fig, 2, i)
            rf_avg = np.mean(results.roc_auc, axis=1)
            rf_sem = sem(results.roc_auc, axis=1)
            x = np.linspace(-self.pre_window, self.event_length + self.post_window, len(rf_avg))
            rf_shuffle_avg = np.mean(results.roc_auc_shuffle, axis=1)
            rf_shuffle_sem = sem(results.roc_auc_shuffle, axis=1)
            plt.plot(x, rf_avg, label="rf")
            plt.fill_between(x, rf_avg - rf_sem, rf_avg + rf_sem, alpha=0.2)
            plt.plot(x, rf_shuffle_avg, label="rf shuffle")
            plt.fill_between(x, rf_shuffle_avg - rf_shuffle_sem, rf_shuffle_avg + rf_shuffle_sem, alpha=0.2)
            plt.title(f"{key}")
            plt.ylim(0.4, 1)
            plt.axvline(x=0, color="k", linestyle="--")
            if i == 2:
                plt.legend(bbox_to_anchor=(1, 1))
            i += 1
        plt.suptitle("Decoder Accuracy")
        plt.show()

    def plot_average(self, start=0, stop=None):
        no_plots = len(self.events)
        height_fig = math.ceil(no_plots / 2)
        i = 1
        bar_width = 0.2
        total_event = self.event_length + self.post_window
        plt.figure(figsize=(8, 4 * height_fig))
        for key, results in self.results.items():
            plt.subplot(height_fig, 2, i)
            x = np.linspace(-self.pre_window, total_event, np.array(results.roc_auc).shape[0])
            if start is not None:
                start = np.where(x >= start)[0][0]
            if stop is None:
                stop = results.roc_auc.shape[0]
            if stop is not None:
                stop = np.where(x <= stop)[0][-1] + 1
            rf_avg = np.mean(np.mean(results.roc_auc[start:stop], axis=0), axis=0)
            rf_sem = sem(np.mean(results.roc_auc[start:stop], axis=0))
            rf_shuffle_avg = np.mean(np.mean(results.roc_auc_shuffle[start:stop], axis=0), axis=0)
            rf_shuffle_sem = sem(np.mean(results.roc_auc_shuffle[start:stop], axis=0))
            bar_positions = np.array([0.3, 0.6])
            plt.bar(bar_positions[0], rf_avg, bar_width, label="RF", yerr=rf_sem, capsize=5)
            plt.bar(bar_positions[1], rf_shuffle_avg, bar_width, label="RF Shuffle", yerr=rf_shuffle_sem, capsize=5)
            plt.title(f"{key}")
            plt.ylim(0.4, 1)
            if i == 2:
                plt.legend(bbox_to_anchor=(1, 1))
            i += 1
            plt.xticks([])
        plt.suptitle("Decoder Accuracy")
        plt.show()


class model_results:
    def __init__(self, model_dict, shuffle_dict, labels, num_fold):
        self.total_trials = len(labels)
        self.reconfig_data(model_dict, num_fold)
        self.reconfig_data(shuffle_dict, num_fold, shuffle=True)

    def reconfig_data(self, model_dict, num_fold, shuffle=False):
        models = []
        timebins = len(model_dict)
        roc_auc = np.empty([timebins, num_fold])
        if not shuffle:
            probabilities = []
            labels = []
        for i in range(timebins):
            roc_auc[i] = model_dict[i]["test_roc_auc"]
            if not shuffle:
                models.append(model_dict[i]["estimator"])
                probabilities_for_t = model_dict[i]["probabilities"]["probabilities"]
                labels_for_t = model_dict[i]["probabilities"]["labels"]
                probabilities.append(probabilities_for_t)
                labels.append(labels_for_t)
        if not shuffle:
            # probabilities = [timebins, folds, classes]
            self.probabilities = probabilities
            # labels = [timebins, folds, trials]
            self.labels = labels
            # models = [timebins, folds]
            self.models = models
            # roc_auc = [timebins, folds]
            self.roc_auc = roc_auc
            self.avg_auc = np.mean(np.mean(roc_auc, axis=0), axis=0)
        if shuffle:
            self.roc_auc_shuffle = roc_auc
            self.avg_shuffle_auc = np.mean(np.mean(roc_auc, axis=0), axis=0)

    def __repr__(self):
        output = ["Model Results"]
        output.append(f"Average AUC score: {self.avg_auc}")
        output.append(f"Average AUC score for shuffled data: {self.avg_shuffle_auc}")
        # output.append(f"Total positive trials:{self.pos_labels}: Total neg trials:{self.neg_labels}")
        return "\n".join(output)


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


def top_feat_trial_decoder(
    num_fold, num_shuffle, events, model, top_features=None, baseline=None, event_len=None, pre_window=0, post_window=0
):
    power_data = test_analysis.average_events(
        events=events,
        mode="power",
        baseline=baseline,
        event_len=event_len,
        pre_window=pre_window,
        post_window=post_window,
        plot=False,
    )
    coherence_data = test_analysis.average_events(
        events=events,
        mode="coherence",
        baseline=baseline,
        event_len=event_len,
        pre_window=pre_window,
        post_window=post_window,
        plot=False,
    )
    # for not granger: data = [trials, brain regions/pairs, frequencies]
    # for granger: data = [trials, brain region, brain region, freqeuncies]
    [power_agent_band_dict, power_band_agent_dict] = lfp.band_calcs(power_data)
    [coherence_agent_band_dict, coherence_band_agent_dict] = lfp.band_calcs(coherence_data)
    # decoder data = [trials, ...] for each band
    decoder_data = {}
    if top_features is not None:
        power_indices, coherence_indices = get_feature_indices(top_features)
    for event in events:
        p_band_dict = power_agent_band_dict[event]
        c_band_dict = coherence_agent_band_dict[event]
        p_stacked_bands = np.stack(list(p_band_dict.values()), axis=0)
        c_stacked_bands = np.stack(list(c_band_dict.values()), axis=0)
        # stacked bands = [band, trials, ...]
        p_reshaped_bands = np.transpose(p_stacked_bands, (1, 0, 2))
        c_reshaped_bands = np.transpose(c_stacked_bands, (1, 0, 2))
        p_bands = p_reshaped_bands.reshape(p_reshaped_bands.shape[0], -1)
        c_bands = c_reshaped_bands.reshape(c_reshaped_bands.shape[0], -1)
        if top_features is not None:
            p_bands = p_bands[:, power_indices]
            c_bands = c_bands[:, coherence_indices]
        pc_bands = np.concatenate([p_bands, c_bands], axis=1)
        decoder_data[event] = pc_bands
    auc = {}
    prob = {}
    weights = {}
    for event in events:
        data_neg = []
        data_pos = []
        for trial in range(decoder_data[event].shape[0]):
            trial_data = decoder_data[event][trial, ...]
            # if np.sum(np.isnan(trial_data)) == 0:
            data_pos.append(trial_data)
        for neg_event in np.setdiff1d(events, event):
            for trial in range(decoder_data[neg_event].shape[0]):
                trial_data = decoder_data[neg_event][trial, ...]
                #    if np.sum(np.isnan(trial_data)) == 0:
                data_neg.append(trial_data)
        data_pos = np.stack(data_pos)
        # data_pos = trials x features
        data_neg = np.stack(data_neg)
        num_pos = data_pos.shape[0]
        num_neg = data_neg.shape[0]
        print(num_pos + num_neg)
        data_pos = data_pos[np.random.permutation(num_pos), :]
        data_neg = data_neg[np.random.permutation(num_neg), :]
        model_keys = {
            "glm": ["glm", "glm_shuffle"],
            "rf": ["rf", "rf_shuffle"],
            "both": ["glm", "rf", "glm_shuffle", "rf_shuffle"],
        }
        auc[event] = {key: [] for key in model_keys[model]}
        prob[event] = {key: [] for key in model_keys[model]}
        weights[event] = {key: [] for key in model_keys[model]}
        for fold in range(num_fold):
            data_test, label_test, data_train, label_train = __train_test_split__(
                fold, num_fold, data_pos, data_neg, num_pos, num_neg
            )
            pred_glm, weight_glm, pred_rf, feat_imp_rf = __run_model__(model, data_train, data_test, label_train)
            if (model == "glm") | (model == "both"):
                auc_glm = roc_auc_score(label_test, pred_glm[:, 1])
                auc[event]["glm"].append(auc_glm)
                prob[event]["glm"].append(pred_glm)
                weights[event]["glm"].append(weight_glm)
            if (model == "rf") | (model == "both"):
                auc_rf = roc_auc_score(label_test, pred_rf[:, 1])
                auc[event]["rf"].append(auc_rf)
                prob[event]["rf"].append(pred_rf)
                weights[event]["rf"].append(feat_imp_rf)
        for shuffle in range(num_shuffle):
            label_train = np.random.permutation(label_train)
            pred_glm, weight_glm, pred_rf, feat_imp_rf = __run_model__(model, data_train, data_test, label_train)
            if (model == "rf") | (model == "both"):
                auc_rf = roc_auc_score(label_test, pred_rf[:, 1])
                auc[event]["rf_shuffle"].append(auc_rf)
                prob[event]["rf_shuffle"].append(pred_rf)
                weights[event]["rf_shuffle"].append(feat_imp_rf)
            if (model == "glm") | (model == "both"):
                auc_glm = roc_auc_score(label_test, pred_glm[:, 1])
                auc[event]["glm_shuffle"].append(auc_glm)
                prob[event]["glm_shuffle"].append(pred_glm)
                weights[event]["glm_shuffle"].append(weight_glm)
    return [auc, prob, weights]
