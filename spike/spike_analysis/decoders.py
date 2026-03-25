import numpy as np
import spike.spike_analysis.pca_trajectories as pca_traj
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
import matplotlib.pyplot as plt
import math
from scipy.stats import sem
import pandas as pd
from collections import defaultdict


def trial_PCA(
    spike_collection, event_length, pre_window, post_window, no_PCs, events, min_neurons=0, condition_dict=None,
    return_recording_labels=False
):
    if condition_dict is None:
        pc_result = pca_traj.avg_trajectories_pca(
            spike_collection,
            event_length,
            pre_window,
            post_window,
            events=events,
            min_neurons=min_neurons,
            plot=False,
        )
    else:
        pc_result = pca_traj.coniditon_pca(
            spike_collection,
            condition_dict,
            event_length,
            pre_window,
            post_window,
            events=events,
            min_neurons=min_neurons,
            plot=False,
        )
    full_PCA_matrix = pc_result.matrix_df
    # time bins by neurons
    # coefficients = components x features (PCs x neurons)
    coefficients = pc_result.coefficients
    recordings = full_PCA_matrix.columns.to_list()
    coefficients = coefficients[:, :no_PCs]
    coefficients_df = pd.DataFrame(data=coefficients, index=recordings)
    decoder_data = defaultdict(list)
    recording_labels = defaultdict(list)
    if condition_dict is not None:
        recording_to_condition = {rec: cond for cond, recs in condition_dict.items() for rec in recs}
    # decoder data dict: events for keys, values is a list of len(events)
    # each element in the list is the transformed matrix
    for recording in spike_collection.recordings:
        # trim weight matrix for only those neurons in the current recording
        try:
            subset_coeff = coefficients_df.loc[[recording.name]]
            for event in events:
                if condition_dict is not None:
                    event_name = recording_to_condition[recording.name] + " " + event
                else:
                    event_name = event
                # grab all event firing rates for current event in current recording
                event_firing_rates = recording.event_firing_rates(event, event_length, pre_window, post_window)
                for trial in range(len(event_firing_rates)):
                    # iterate through each event
                    trial_data = np.dot(event_firing_rates[trial], subset_coeff)
                    # transform each trial with original weight matrix
                    # T (timebins x pcs) = D (timebins x neurons). W (pcs x neurons)
                    decoder_data[event_name].append(trial_data)
                    recording_labels[event_name].append(recording.name)
        except KeyError:
            pass
    if return_recording_labels:
        return decoder_data, recording_labels
    return decoder_data


def trial_decoder(
    spike_collection,
    num_fold,
    no_PCs,
    events,
    event_length,
    pre_window=0,
    post_window=0,
    min_neurons=0,
    condition_dict=None,
    decoder_data=None,
    plot=True,
    LOO=False,
):
    if LOO:
        return _trial_decoder_LOO(
            spike_collection, no_PCs, events, event_length, pre_window, post_window, min_neurons, condition_dict
        )
    if decoder_data is None:
        decoder_data = trial_PCA(
            spike_collection,
            event_length,
            pre_window,
            post_window,
            no_PCs,
            events=events,
            min_neurons=min_neurons,
            condition_dict=condition_dict,
        )
    T = decoder_data[events[0]][0].shape[0]
    results_dict = {}
    shuffle_results_dict = {}
    event_labels = {}
    for event in events:
        results_dict[event] = []
        shuffle_results_dict[event] = []
        data, labels = __prep_data__(decoder_data, events, event)
        event_labels[event] = labels
        # data = [timebins, pcs, trials]
        for timebin in range(T):
            t_data = data[:, :, timebin]
            rf = BaggingClassifier(
                estimator=DecisionTreeClassifier(class_weight="balanced"), n_estimators=100, random_state=0, n_jobs=-1
            )
            results = cross_validate(
                rf,
                t_data,
                labels,
                scoring=["roc_auc"],
                cv=num_fold,
                n_jobs=-1,
                return_estimator=True,
                return_indices=True,
            )
            shuffle_results = cross_validate(
                rf,
                t_data,
                np.random.permutation(labels),
                scoring=["roc_auc"],
                cv=num_fold,
                n_jobs=-1,
            )
            prob_dict = __probabilities__(results, labels, t_data, num_fold)
            results["probabilities"] = prob_dict
            results_dict[event].append(results)
            shuffle_results_dict[event].append(shuffle_results)
        if len(events) == 2:
            break
    result_object = all_results(
        results_dict, shuffle_results_dict, num_fold, event_labels, event_length, pre_window, post_window
    )
    return result_object


def _trial_decoder_LOO(spike_collection, no_PCs, events, event_length, pre_window, post_window, min_neurons, condition_dict):
    decoder_data, recording_labels = trial_PCA(
        spike_collection, event_length, pre_window, post_window, no_PCs,
        events=events, min_neurons=min_neurons, condition_dict=condition_dict,
        return_recording_labels=True,
    )
    T = decoder_data[events[0]][0].shape[0]
    # collect unique recording names in a stable order
    all_recording_names = list(dict.fromkeys(
        name for event in recording_labels for name in recording_labels[event]
    ))
    N = len(all_recording_names)

    results_dict = {}
    shuffle_results_dict = {}
    event_labels = {}

    for event in events:
        results_dict[event] = []
        shuffle_results_dict[event] = []
        data, labels, rec_label_arr = __prep_data_loo__(decoder_data, recording_labels, events, event)
        event_labels[event] = labels

        for timebin in range(T):
            t_data = data[:, :, timebin]
            fold_aucs = []
            fold_shuffle_aucs = []
            fold_models = []
            fold_probs = []
            fold_prob_labels = []

            for rec_name in all_recording_names:
                test_mask = rec_label_arr == rec_name
                train_mask = ~test_mask
                X_train, y_train = t_data[train_mask], labels[train_mask]
                X_test, y_test = t_data[test_mask], labels[test_mask]

                # skip fold if either split is missing a class
                if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
                    continue

                rf = BaggingClassifier(
                    estimator=DecisionTreeClassifier(class_weight="balanced"),
                    n_estimators=100, random_state=0, n_jobs=-1,
                )
                rf.fit(X_train, y_train)
                y_prob = rf.predict_proba(X_test)[:, 1]
                fold_aucs.append(roc_auc_score(y_test, y_prob))
                fold_models.append(rf)
                fold_probs.append(rf.predict_proba(X_test))
                fold_prob_labels.append(y_test)

                rf_shuf = BaggingClassifier(
                    estimator=DecisionTreeClassifier(class_weight="balanced"),
                    n_estimators=100, random_state=0, n_jobs=-1,
                )
                rf_shuf.fit(X_train, np.random.permutation(y_train))
                y_prob_shuf = rf_shuf.predict_proba(X_test)[:, 1]
                fold_shuffle_aucs.append(roc_auc_score(y_test, y_prob_shuf))

            results_dict[event].append({
                "test_roc_auc": np.array(fold_aucs),
                "estimator": fold_models,
                "probabilities": {"probabilities": fold_probs, "labels": fold_prob_labels},
            })
            shuffle_results_dict[event].append({"test_roc_auc": np.array(fold_shuffle_aucs)})

        if len(events) == 2:
            break

    return all_results(results_dict, shuffle_results_dict, N, event_labels, event_length, pre_window, post_window)


def __prep_data_loo__(decoder_data, recording_labels, events, event):
    data_pos, data_neg = [], []
    rec_pos, rec_neg = [], []
    for trial, rec in zip(decoder_data[event], recording_labels[event]):
        data_pos.append(trial)
        rec_pos.append(rec)
    for neg_event in np.setdiff1d(events, event):
        for trial, rec in zip(decoder_data[neg_event], recording_labels[neg_event]):
            data_neg.append(trial)
            rec_neg.append(rec)
    data_pos = np.stack(data_pos, axis=2)
    data_neg = np.stack(data_neg, axis=2)
    label_pos = np.ones(data_pos.shape[2])
    label_neg = np.zeros(data_neg.shape[2])
    all_data = np.concatenate([data_pos, data_neg], axis=2)
    data = all_data.transpose()  # (trials, PCs, timebins)
    labels = np.concatenate([label_pos, label_neg])
    rec_label_arr = np.array(rec_pos + rec_neg)
    shuffle = np.random.permutation(len(labels))
    return data[shuffle], labels[shuffle], rec_label_arr[shuffle]


def __prep_data__(decoder_data, events, event):
    data_neg = []
    data_pos = []
    for trial in decoder_data[event]:
        data_pos.append(trial)
    for neg_event in np.setdiff1d(events, event):
        for trial in decoder_data[neg_event]:
            data_neg.append(trial)
    data_pos = np.stack(data_pos, axis=2)
    data_neg = np.stack(data_neg, axis=2)
    label_pos = np.ones(data_pos.shape[2])
    label_neg = np.zeros(data_neg.shape[2])
    all_data = np.concatenate([data_pos, data_neg], axis=2)
    # data = (samples, features, timebins)
    data = all_data.transpose()
    labels = np.concatenate([label_pos, label_neg], axis=0)
    shuffle = np.random.permutation(len(labels))
    data = data[shuffle, :, :]
    labels = labels[shuffle]
    return data, labels


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
        self.events = list(results_dict.keys())
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
                plot_start = np.where(x >= start)[0][0]
            if stop is None:
                plot_stop = results.roc_auc.shape[0]
            if stop is not None:
                plot_stop = np.where(x <= stop)[0][-1] + 1
            rf_avg = np.mean(np.mean(results.roc_auc[start:stop], axis=0), axis=0)
            rf_sem = sem(np.mean(results.roc_auc[plot_start:plot_stop], axis=0))
            rf_shuffle_avg = np.mean(np.mean(results.roc_auc_shuffle[plot_start:plot_stop], axis=0), axis=0)
            rf_shuffle_sem = sem(np.mean(results.roc_auc_shuffle[plot_start:plot_stop], axis=0))
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
