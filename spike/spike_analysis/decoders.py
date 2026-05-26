import numpy as np
import spike.spike_analysis.population_analysis as pca_traj
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_validate
import matplotlib.pyplot as plt
import math
from scipy.stats import sem
import pandas as pd
from collections import defaultdict
from itertools import permutations as iter_perms, combinations


def trial_PCA(
    spike_collection, event_length, pre_window, post_window, percent_var=90, events=None,
    min_neurons=0, condition_dict=None, return_recording_labels=False, no_PCs=None
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
        pc_result = pca_traj.condition_pca(
            spike_collection,
            condition_dict,
            event_length,
            pre_window,
            post_window,
            events=events,
            min_neurons=min_neurons,
            plot=False,
        )
    if no_PCs is None:
        no_PCs = pca_traj.PCs_needed(pc_result.explained_variance, percent_var / 100)
    num_pcs = no_PCs
    full_PCA_matrix = pc_result.matrix_df
    recordings = full_PCA_matrix.columns.to_list()
    # precompute each recording's unit positions in the full matrix
    recordings_arr = np.array(recordings)
    neuron_indices = {
        rec: np.where(recordings_arr == rec)[0]
        for rec in np.unique(recordings_arr)
    }
    scaler_mean = pc_result.scaler.mean_
    scaler_scale = pc_result.scaler.scale_
    # W = coefficients [n_PCs, n_units]; subset for recording i = W[:no_PCs, unit_idx].T → [n_i, no_PCs]
    W = pc_result.coefficients
    decoder_data = defaultdict(list)
    recording_labels = defaultdict(list)
    if condition_dict is not None:
        recording_to_condition = {rec: cond for cond, recs in condition_dict.items() for rec in recs}
    for recording in spike_collection.recordings:
        try:
            idx = neuron_indices[recording.name]
            subset_coeff = W[:no_PCs, idx].T        # [n_i, no_PCs]
            rec_mean = scaler_mean[idx]
            rec_scale = scaler_scale[idx]
            for event in events:
                if condition_dict is not None:
                    event_name = recording_to_condition[recording.name] + " " + event
                else:
                    event_name = event
                event_firing_rates = recording.event_firing_rates(event, event_length, pre_window, post_window)
                for trial in range(len(event_firing_rates)):
                    normalized_trial = (event_firing_rates[trial] - rec_mean) / rec_scale
                    trial_data = np.dot(normalized_trial, subset_coeff)
                    decoder_data[event_name].append(trial_data)
                    recording_labels[event_name].append(recording.name)
        except KeyError:
            pass
    if return_recording_labels:
        return decoder_data, recording_labels, num_pcs
    return decoder_data, num_pcs


def _random_forest(t_data, labels, num_fold):
    """k-fold RF decoder for one timebin."""
    rf = BaggingClassifier(
        estimator=DecisionTreeClassifier(class_weight="balanced"),
        n_estimators=100, random_state=0, n_jobs=-1,
    )
    results = cross_validate(
        rf, t_data, labels,
        scoring=["roc_auc"], cv=num_fold, n_jobs=-1,
        return_estimator=True, return_indices=True,
    )
    shuffle_results = cross_validate(
        rf, t_data, np.random.permutation(labels),
        scoring=["roc_auc"], cv=num_fold, n_jobs=-1,
    )
    results["probabilities"] = __probabilities__(results, labels, t_data, num_fold)
    return results, shuffle_results


def _random_forest_fold(X_train, y_train, X_test, y_test):
    """Single LOO fold for RF decoder. Returns (auc, auc_shuffle, model, probs, labels)."""
    rf = BaggingClassifier(
        estimator=DecisionTreeClassifier(class_weight="balanced"),
        n_estimators=100, random_state=0, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    rf_shuf = BaggingClassifier(
        estimator=DecisionTreeClassifier(class_weight="balanced"),
        n_estimators=100, random_state=0, n_jobs=-1,
    )
    rf_shuf.fit(X_train, np.random.permutation(y_train))
    auc_shuf = roc_auc_score(y_test, rf_shuf.predict_proba(X_test)[:, 1])

    return auc, auc_shuf, rf, rf.predict_proba(X_test), y_test


def _linear(t_data, labels, num_fold, C=1, **kwargs):
    """k-fold LinearSVC decoder for one timebin. Uses decision_function for ROC AUC."""
    clf = LinearSVC(
        class_weight="balanced",
        C=C,
        max_iter=5000,  # matching decodanda defaults
        dual=False,     # matching decodanda defaults
        **kwargs,
    )
    results = cross_validate(
        clf, t_data, labels,
        scoring=["roc_auc"], cv=num_fold, n_jobs=-1,
        return_estimator=True, return_indices=True,
    )
    shuffle_results = cross_validate(
        clf, t_data, np.random.permutation(labels),
        scoring=["roc_auc"], cv=num_fold, n_jobs=-1,
    )
    results["probabilities"] = __decision_scores__(results, labels, t_data, num_fold)
    return results, shuffle_results


def _linear_fold(X_train, y_train, X_test, y_test, C=1, **kwargs):
    """Single LOO fold for LinearSVC decoder. Returns (auc, auc_shuffle, model, scores, labels)."""
    clf = LinearSVC(
        class_weight="balanced",
        C=C,
        max_iter=5000,  # matching decodanda defaults
        dual=False,     # matching decodanda defaults
        **kwargs,
    )
    clf.fit(X_train, y_train)
    scores = clf.decision_function(X_test)
    auc = roc_auc_score(y_test, scores)

    clf_shuf = LinearSVC(
        class_weight="balanced",
        C=C,
        max_iter=5000,  # matching decodanda defaults
        dual=False,     # matching decodanda defaults
        **kwargs,
    )
    clf_shuf.fit(X_train, np.random.permutation(y_train))
    auc_shuf = roc_auc_score(y_test, clf_shuf.decision_function(X_test))

    return auc, auc_shuf, clf, scores, y_test


def _flat_decoder(spike_collection, num_fold, events, event_length, pre_window, post_window,
                  min_neurons, condition_dict, decoder_data, classifier_type, percent_var=90, no_PCs=None, **kwargs):
    """Run one classifier per event on full flattened trial (T * n_PCs features) instead of per timebin."""
    if decoder_data is None:
        decoder_data, num_pcs = trial_PCA(
            spike_collection, event_length, pre_window, post_window,
            percent_var=percent_var, events=events, min_neurons=min_neurons,
            condition_dict=condition_dict, no_PCs=no_PCs,
        )
    else:
        num_pcs = no_PCs
    results_dict = {}
    shuffle_results_dict = {}
    event_labels = {}
    for event in events:
        results_dict[event] = []
        shuffle_results_dict[event] = []
        data, labels = __prep_data_flat__(decoder_data, events, event)
        event_labels[event] = labels
        if classifier_type == "linear":
            results, shuffle_results = _linear(data, labels, num_fold, **kwargs)
        else:
            results, shuffle_results = _random_forest(data, labels, num_fold)
        results_dict[event].append(results)
        shuffle_results_dict[event].append(shuffle_results)
        if len(events) == 2:
            break
    return all_results(results_dict, shuffle_results_dict, num_fold, event_labels, event_length, pre_window, post_window, num_pcs=num_pcs)


def trial_decoder(
    spike_collection,
    num_fold,
    events,
    event_length,
    percent_var=90,
    pre_window=0,
    post_window=0,
    min_neurons=0,
    condition_dict=None,
    decoder_data=None,
    plot=True,
    LOO=False,
    input="timebin",
    classifier_type="RF",
    no_PCs=None,
    **kwargs,
):
    if LOO:
        return _trial_decoder_LOO(
            spike_collection, events, event_length, pre_window, post_window, min_neurons, condition_dict,
            percent_var=percent_var, no_PCs=no_PCs, classifier_type=classifier_type, input=input, **kwargs,
        )
    if input == "full_trial":
        return _flat_decoder(
            spike_collection, num_fold, events, event_length, pre_window, post_window,
            min_neurons, condition_dict, decoder_data, classifier_type,
            percent_var=percent_var, no_PCs=no_PCs, **kwargs,
        )
    if decoder_data is None:
        decoder_data, num_pcs = trial_PCA(
            spike_collection, event_length, pre_window, post_window,
            percent_var=percent_var, events=events, min_neurons=min_neurons,
            condition_dict=condition_dict, no_PCs=no_PCs,
        )
    else:
        num_pcs = no_PCs
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
            if classifier_type == "linear":
                results, shuffle_results = _linear(t_data, labels, num_fold, **kwargs)
            else:
                results, shuffle_results = _random_forest(t_data, labels, num_fold)
            results_dict[event].append(results)
            shuffle_results_dict[event].append(shuffle_results)
        if len(events) == 2:
            break
    result_object = all_results(
        results_dict, shuffle_results_dict, num_fold, event_labels, event_length, pre_window, post_window,
        num_pcs=num_pcs, percent_var=percent_var,
    )
    return result_object


def _trial_decoder_LOO(spike_collection, events, event_length, pre_window, post_window, min_neurons, condition_dict, percent_var=90, no_PCs=None, classifier_type="RF", input="timebin", decoder_data=None, recording_labels=None, **kwargs):
    if decoder_data is None or recording_labels is None:
        decoder_data, recording_labels, num_pcs = trial_PCA(
            spike_collection, event_length, pre_window, post_window,
            percent_var=percent_var, events=events, min_neurons=min_neurons,
            condition_dict=condition_dict, return_recording_labels=True, no_PCs=no_PCs,
        )
    else:
        num_pcs = no_PCs
    # collect unique recording names in a stable order
    all_recording_names = list(dict.fromkeys(
        name for event in recording_labels for name in recording_labels[event]
    ))
    N = len(all_recording_names)

    results_dict = {}
    shuffle_results_dict = {}
    event_labels = {}

    if input == "full_trial":
        for event in events:
            results_dict[event] = []
            shuffle_results_dict[event] = []
            data, labels, rec_label_arr = __prep_data_flat_loo__(decoder_data, recording_labels, events, event)
            event_labels[event] = labels
            fold_aucs, fold_shuffle_aucs, fold_models, fold_probs, fold_prob_labels = [], [], [], [], []
            for rec_name in all_recording_names:
                test_mask = rec_label_arr == rec_name
                train_mask = ~test_mask
                X_train, y_train = data[train_mask], labels[train_mask]
                X_test, y_test = data[test_mask], labels[test_mask]
                if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
                    continue
                if classifier_type == "linear":
                    auc, auc_shuf, model, probs, prob_labels = _linear_fold(X_train, y_train, X_test, y_test, **kwargs)
                else:
                    auc, auc_shuf, model, probs, prob_labels = _random_forest_fold(X_train, y_train, X_test, y_test)
                fold_aucs.append(auc)
                fold_shuffle_aucs.append(auc_shuf)
                fold_models.append(model)
                fold_probs.append(probs)
                fold_prob_labels.append(prob_labels)
            results_dict[event].append({
                "test_roc_auc": np.array(fold_aucs),
                "estimator": fold_models,
                "probabilities": {"probabilities": fold_probs, "labels": fold_prob_labels},
            })
            shuffle_results_dict[event].append({"test_roc_auc": np.array(fold_shuffle_aucs)})
            if len(events) == 2:
                break
    else:
        T = decoder_data[events[0]][0].shape[0]
        for event in events:
            results_dict[event] = []
            shuffle_results_dict[event] = []
            data, labels, rec_label_arr = __prep_data_loo__(decoder_data, recording_labels, events, event)
            event_labels[event] = labels
            for timebin in range(T):
                t_data = data[:, :, timebin]
                fold_aucs, fold_shuffle_aucs, fold_models, fold_probs, fold_prob_labels = [], [], [], [], []
                for rec_name in all_recording_names:
                    test_mask = rec_label_arr == rec_name
                    train_mask = ~test_mask
                    X_train, y_train = t_data[train_mask], labels[train_mask]
                    X_test, y_test = t_data[test_mask], labels[test_mask]
                    if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
                        continue
                    if classifier_type == "linear":
                        auc, auc_shuf, model, probs, prob_labels = _linear_fold(X_train, y_train, X_test, y_test, **kwargs)
                    else:
                        auc, auc_shuf, model, probs, prob_labels = _random_forest_fold(X_train, y_train, X_test, y_test)
                    fold_aucs.append(auc)
                    fold_shuffle_aucs.append(auc_shuf)
                    fold_models.append(model)
                    fold_probs.append(probs)
                    fold_prob_labels.append(prob_labels)
                results_dict[event].append({
                    "test_roc_auc": np.array(fold_aucs),
                    "estimator": fold_models,
                    "probabilities": {"probabilities": fold_probs, "labels": fold_prob_labels},
                })
                shuffle_results_dict[event].append({"test_roc_auc": np.array(fold_shuffle_aucs)})
            if len(events) == 2:
                break

    return all_results(results_dict, shuffle_results_dict, N, event_labels, event_length, pre_window, post_window, num_pcs=num_pcs, percent_var=percent_var)


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


def __prep_data_flat_loo__(decoder_data, recording_labels, events, event):
    """Like __prep_data_loo__ but flattens each trial (T, n_PCs) → (T*n_PCs,) before stacking."""
    data_pos, data_neg = [], []
    rec_pos, rec_neg = [], []
    for trial, rec in zip(decoder_data[event], recording_labels[event]):
        data_pos.append(trial.flatten())
        rec_pos.append(rec)
    for neg_event in np.setdiff1d(events, event):
        for trial, rec in zip(decoder_data[neg_event], recording_labels[neg_event]):
            data_neg.append(trial.flatten())
            rec_neg.append(rec)
    data_pos = np.stack(data_pos)   # (n_pos, T*n_PCs)
    data_neg = np.stack(data_neg)   # (n_neg, T*n_PCs)
    label_pos = np.ones(len(data_pos))
    label_neg = np.zeros(len(data_neg))
    all_data = np.concatenate([data_pos, data_neg], axis=0)  # (trials, T*n_PCs)
    labels = np.concatenate([label_pos, label_neg])
    rec_label_arr = np.array(rec_pos + rec_neg)
    shuffle = np.random.permutation(len(labels))
    return all_data[shuffle], labels[shuffle], rec_label_arr[shuffle]


def __prep_data_flat__(decoder_data, events, event):
    """Like __prep_data__ but flattens each trial (T, n_PCs) → (T*n_PCs,) before stacking."""
    data_pos, data_neg = [], []
    for trial in decoder_data[event]:
        data_pos.append(trial.flatten())
    for neg_event in np.setdiff1d(events, event):
        for trial in decoder_data[neg_event]:
            data_neg.append(trial.flatten())
    data_pos = np.stack(data_pos)
    data_neg = np.stack(data_neg)
    label_pos = np.ones(len(data_pos))
    label_neg = np.zeros(len(data_neg))
    all_data = np.concatenate([data_pos, data_neg], axis=0)
    labels = np.concatenate([label_pos, label_neg])
    shuffle = np.random.permutation(len(labels))
    return all_data[shuffle], labels[shuffle]


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


def __decision_scores__(results, labels, t_data, num_fold):
    scores = []
    score_labels = []
    for i in range(num_fold):
        test_indices = results["indices"]["test"][i]
        test_data = t_data[test_indices, :]
        test_labels = labels[test_indices]
        model = results["estimator"][i]
        s = model.decision_function(test_data)
        scores.append(s)
        score_labels.append(test_labels)
    return {"probabilities": scores, "labels": score_labels}


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


def _within_subject_shuffle(decoder_data, recording_labels, events, perm_type="random"):
    """Permute event labels at the recording level.

    perm_type="random"  : each recording independently gets a random non-identity permutation.
    perm_type="balanced": permutations are shuffled then cycled across recordings so no two
                          recordings get the same assignment (as much as possible).
    """
    all_recordings = list(dict.fromkeys(
        rec for event in events for rec in recording_labels[event]
    ))
    # all non-identity permutations of events
    all_perms = [p for p in iter_perms(events) if list(p) != list(events)]

    if perm_type == "balanced":
        # shuffle permutation order so assignment isn't always the same across runs
        perm_order = list(np.random.permutation(len(all_perms)))
        assigned_perms = {rec: all_perms[perm_order[i % len(all_perms)]] for i, rec in enumerate(all_recordings)}
    else:  # random
        assigned_perms = {rec: all_perms[np.random.randint(len(all_perms))] for rec in all_recordings}

    new_decoder_data = {event: [] for event in events}
    new_recording_labels = {event: [] for event in events}

    for rec in all_recordings:
        perm = assigned_perms[rec]  # e.g. ("event_b", "event_c", "event_a")
        for orig_event, new_event in zip(events, perm):
            for trial, rec_label in zip(decoder_data[orig_event], recording_labels[orig_event]):
                if rec_label == rec:
                    new_decoder_data[new_event].append(trial)
                    new_recording_labels[new_event].append(rec_label)

    return new_decoder_data, new_recording_labels


def trial_decoder_within_subject_shuffle(
    spike_collection,
    num_fold,
    events,
    event_length,
    percent_var=90,
    pre_window=0,
    post_window=0,
    min_neurons=0,
    condition_dict=None,
    plot=True,
    LOO=False,
    input="timebin",
    classifier_type="RF",
    perm_type="random",
    no_PCs=None,
    **kwargs,
):
    decoder_data, recording_labels, num_pcs = trial_PCA(
        spike_collection, event_length, pre_window, post_window,
        percent_var=percent_var, events=events, min_neurons=min_neurons,
        condition_dict=condition_dict, return_recording_labels=True, no_PCs=no_PCs,
    )
    shuffled_data, shuffled_labels = _within_subject_shuffle(
        decoder_data, recording_labels, events, perm_type
    )
    if LOO:
        return _trial_decoder_LOO(
            spike_collection, events, event_length, pre_window, post_window,
            min_neurons, condition_dict,
            percent_var=percent_var, no_PCs=num_pcs,
            classifier_type=classifier_type, input=input,
            decoder_data=shuffled_data, recording_labels=shuffled_labels,
            **kwargs,
        )
    return trial_decoder(
        spike_collection, num_fold, events, event_length,
        percent_var=percent_var, pre_window=pre_window, post_window=post_window,
        min_neurons=min_neurons, condition_dict=condition_dict, decoder_data=shuffled_data,
        plot=plot, LOO=False, input=input, classifier_type=classifier_type,
        no_PCs=num_pcs, **kwargs,
    )


class all_results:
    def __init__(self, results_dict, shuffle_dict, num_fold, event_labels, event_length, pre_window, post_window, num_pcs=None, percent_var=None):
        self.num_fold = num_fold
        self.events = list(results_dict.keys())
        self.event_length = event_length
        self.pre_window = pre_window
        self.post_window = post_window
        self.num_pcs = num_pcs
        self.percent_var = percent_var
        results = {}
        for event in self.events:
            results[event] = model_results(results_dict[event], shuffle_dict[event], event_labels[event], num_fold)
        self.results = results

    def __repr__(self):
        output = [f"Models ran with {self.num_fold} folds"]
        if self.percent_var is not None:
            output.append(f"Variance threshold: {self.percent_var}%  |  PCs used: {self.num_pcs}")
        elif self.num_pcs is not None:
            output.append(f"PCs used: {self.num_pcs}")
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


# ---------------------------------------------------------------------------
# Per-recording decoder
# ---------------------------------------------------------------------------

class _RecordingWrapper:
    """Wraps a single SpikeRecording so it can be passed where a SpikeCollection is expected.
    Any attribute not explicitly set is proxied to the wrapped recording.
    """
    def __init__(self, recording):
        self._recording = recording
        self.recordings = [recording]

    def __getattr__(self, name):
        return getattr(self._recording, name)


class per_recording_results:
    """Return object for trial_decoder_per_recording.

    Attributes
    ----------
    results : dict {recording_name: all_results}
    recordings : list of recording names
    """
    def __init__(self, results_by_recording):
        self.results = results_by_recording
        self.recordings = list(results_by_recording.keys())

    def __repr__(self):
        lines = [f"Per-recording decoder | {len(self.recordings)} recordings"]
        for name, res in self.results.items():
            pcs_str = f"  [{res.percent_var}% var → {res.num_pcs} PCs]" if res.percent_var is not None else (f"  [{res.num_pcs} PCs]" if res.num_pcs is not None else "")
            auc_str = ", ".join(f"{e}: {mr.avg_auc:.3f}" for e, mr in res.results.items())
            lines.append(f"  {name}:{pcs_str} avg AUC = {auc_str}")
        return "\n".join(lines)

    def _collect_auc_across_recordings(self, event):
        """Returns (n_valid_recordings, T) array of per-fold-mean AUC across time.
        Recordings where all AUC values are NaN are skipped.
        """
        curves = []
        for res in self.results.values():
            if event in res.results:
                curve = np.nanmean(res.results[event].roc_auc, axis=1)  # (T,)
                if not np.all(np.isnan(curve)):
                    curves.append(curve)
        return np.array(curves)  # (n_valid, T)

    def _collect_shuffle_across_recordings(self, event):
        curves = []
        for res in self.results.values():
            if event in res.results:
                curve = np.nanmean(res.results[event].roc_auc_shuffle, axis=1)
                if not np.all(np.isnan(curve)):
                    curves.append(curve)
        return np.array(curves)

    def plot_across_time(self, start=None, stop=None):
        """Plot decode curve averaged across recordings (SEM = inter-subject variability)."""
        if not self.results:
            print("No results to plot.")
            return
        sample_res = next(iter(self.results.values()))
        events = sample_res.events
        pre_window = sample_res.pre_window
        event_length = sample_res.event_length
        post_window = sample_res.post_window

        no_plots = len(events)
        height_fig = math.ceil(no_plots / 2)
        if start is None:
            start = -pre_window
        if stop is None:
            stop = event_length + post_window

        plt.figure(figsize=(12, 4 * height_fig))
        for i, event in enumerate(events, 1):
            curves = self._collect_auc_across_recordings(event)
            shuffle_curves = self._collect_shuffle_across_recordings(event)
            if curves.size == 0:
                continue
            x = np.linspace(-pre_window, event_length + post_window, curves.shape[1])
            avg = np.nanmean(curves, axis=0)
            err = sem(curves, axis=0, nan_policy="omit")
            shuf_avg = np.nanmean(shuffle_curves, axis=0)
            shuf_err = sem(shuffle_curves, axis=0, nan_policy="omit")
            plt.subplot(height_fig, 2, i)
            plt.plot(x, avg, label="decoder")
            plt.fill_between(x, avg - err, avg + err, alpha=0.2)
            plt.plot(x, shuf_avg, label="shuffle")
            plt.fill_between(x, shuf_avg - shuf_err, shuf_avg + shuf_err, alpha=0.2)
            plt.axvline(0, color="k", linestyle="--")
            plt.ylim(0.4, 1)
            plt.title(f"{event} (n={len(curves)})")
            if i == 2:
                plt.legend(bbox_to_anchor=(1, 1))
        plt.suptitle("Per-recording decoder (mean ± SEM across recordings)")
        plt.tight_layout()
        plt.show()

    def plot_average(self, start=0, stop=None):
        """Bar plot of epoch-averaged AUC across recordings."""
        if not self.results:
            print("No results to plot.")
            return
        sample_res = next(iter(self.results.values()))
        events = sample_res.events
        pre_window = sample_res.pre_window
        event_length = sample_res.event_length
        post_window = sample_res.post_window

        no_plots = len(events)
        height_fig = math.ceil(no_plots / 2)
        plt.figure(figsize=(8, 4 * height_fig))
        for i, event in enumerate(events, 1):
            curves = self._collect_auc_across_recordings(event)
            shuffle_curves = self._collect_shuffle_across_recordings(event)
            if curves.size == 0:
                continue
            x = np.linspace(-pre_window, event_length + post_window, curves.shape[1])
            start_idx = np.searchsorted(x, start)
            stop_idx = curves.shape[1] if stop is None else np.searchsorted(x, stop)
            epoch_auc = np.nanmean(curves[:, start_idx:stop_idx], axis=1)
            epoch_shuf = np.nanmean(shuffle_curves[:, start_idx:stop_idx], axis=1)
            epoch_auc = epoch_auc[~np.isnan(epoch_auc)]
            epoch_shuf = epoch_shuf[~np.isnan(epoch_shuf)]
            plt.subplot(height_fig, 2, i)
            bar_positions = np.array([0.3, 0.6])
            plt.bar(bar_positions[0], np.mean(epoch_auc), 0.2, yerr=sem(epoch_auc), capsize=5, label="decoder")
            plt.bar(bar_positions[1], np.mean(epoch_shuf), 0.2, yerr=sem(epoch_shuf), capsize=5, label="shuffle")
            plt.title(f"{event} (n={len(epoch_auc)})")
            plt.ylim(0.4, 1)
            plt.xticks([])
            if i == 2:
                plt.legend(bbox_to_anchor=(1, 1))
        plt.suptitle("Per-recording decoder (mean ± SEM across recordings)")
        plt.tight_layout()
        plt.show()


def trial_decoder_per_recording(
    input,
    num_fold,
    events,
    event_length,
    percent_var=90,
    pre_window=0,
    post_window=0,
    min_neurons=0,
    condition_dict=None,
    classifier_type="RF",
    no_PCs=None,
    **kwargs,
):
    """Run trial_decoder independently per recording, each with its own PCA space.

    Args:
        input: SpikeRecording, list of SpikeRecording, or SpikeCollection
        (remaining args match trial_decoder)

    Returns:
        per_recording_results
    """
    import spike.spike_analysis.spike_recording as _rec_module
    if isinstance(input, _rec_module.SpikeRecording):
        recordings = [input]
    elif isinstance(input, list):
        recordings = input
    else:
        recordings = input.recordings

    results = {}
    for recording in recordings:
        if recording.analyzed_neurons < min_neurons:
            print(f"Skipping {recording.name}: {recording.analyzed_neurons} neurons < min_neurons={min_neurons}")
            continue
        try:
            result = trial_decoder(
                _RecordingWrapper(recording),
                num_fold, events, event_length,
                percent_var=percent_var, pre_window=pre_window, post_window=post_window,
                min_neurons=0, condition_dict=condition_dict,
                plot=False, classifier_type=classifier_type,
                no_PCs=no_PCs, **kwargs,
            )
            results[recording.name] = result
        except Exception as e:
            import traceback
            print(f"Skipping {recording.name}: {e}")
            traceback.print_exc()

    if not results:
        raise RuntimeError("All recordings were skipped — check errors above.")
    return per_recording_results(results)


# ---------------------------------------------------------------------------
# Cross-generalization decoder
# ---------------------------------------------------------------------------

def __split_into_folds__(trials, num_fold):
    """Shuffle a list of trial matrices and split into num_fold bins.

    Returns a list of num_fold sublists, each containing trial matrices.
    """
    idx = np.random.permutation(len(trials))
    fold_indices = np.array_split(idx, num_fold)
    return [[trials[i] for i in fold_idx] for fold_idx in fold_indices]


def __fit_clf_single__(X_train, y_train, classifier_type, C=1, **kwargs):
    """Fit and return a single classifier on (X_train, y_train)."""
    if classifier_type == "linear":
        clf = LinearSVC(class_weight="balanced", C=C, max_iter=5000, dual=False, **kwargs)
    else:
        clf = BaggingClassifier(
            estimator=DecisionTreeClassifier(class_weight="balanced"),
            n_estimators=100, random_state=0, n_jobs=-1,
        )
    clf.fit(X_train, y_train)
    return clf


def __score_test__(clf, X_test, y_test, classifier_type):
    """Score a fitted classifier on a generalization test set. Returns roc_auc."""
    if len(np.unique(y_test)) < 2:
        return np.nan
    if classifier_type == "linear":
        scores = clf.decision_function(X_test)
    else:
        scores = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, scores)


def __build_train_fold__(k, event_folds, e1, e2):
    """Concatenate all non-k bins of e1 (label=0) and e2 (label=1) into a training matrix.

    Args:
        k           : fold index to hold out
        event_folds : {event: list of num_fold sublists of trial matrices}
        e1, e2      : event names for positive (0) and negative (1) class

    Returns:
        X_train_all : np.ndarray (n_train, n_PCs, T)
        y_train     : np.ndarray (n_train,) with 0s and 1s
    """
    train_e1 = [t for i, fold in enumerate(event_folds[e1]) if i != k for t in fold]
    train_e2 = [t for i, fold in enumerate(event_folds[e2]) if i != k for t in fold]
    X_train_all = np.concatenate(
        [np.stack(train_e1, axis=2), np.stack(train_e2, axis=2)], axis=2
    ).transpose()  # (n_train, n_PCs, T)
    y_train = np.array([0] * len(train_e1) + [1] * len(train_e2))
    return X_train_all, y_train


def __build_test_fold__(k, event_folds, test_pairs):
    """Build test matrices for each generalization test pair using fold k.

    Args:
        k           : fold index to use as test
        event_folds : {event: list of num_fold sublists of trial matrices}
        test_pairs  : list of (te1, te2, lab1, lab2) tuples

    Returns:
        X_test_per_pair : {test_key: np.ndarray (n_test, n_PCs, T)}
        y_test_per_pair : {test_key: np.ndarray (n_test,)}
    """
    X_test_per_pair = {}
    y_test_per_pair = {}
    for te1, te2, lab1, lab2 in test_pairs:
        test_key = f"{te1}_{te2}"
        fold_te1 = event_folds[te1][k]
        fold_te2 = event_folds[te2][k]
        X_test_per_pair[test_key] = np.concatenate(
            [np.stack(fold_te1, axis=2), np.stack(fold_te2, axis=2)], axis=2
        ).transpose()  # (n_test, n_PCs, T)
        y_test_per_pair[test_key] = np.array([lab1] * len(fold_te1) + [lab2] * len(fold_te2))
    return X_test_per_pair, y_test_per_pair


def _trial_PCA_holdout(
    spike_collection, event_length, pre_window, post_window,
    pca_events, project_events, percent_var=90, min_neurons=0, condition_dict=None, no_PCs=None,
    return_recording_labels=False,
):
    """Fit PCA on pca_events only, then project all project_events into that space."""
    if condition_dict is not None:
        raise NotImplementedError("hold_out=True is not yet supported with condition_dict")
    pc_result = pca_traj.avg_trajectories_pca(
        spike_collection, event_length, pre_window, post_window,
        events=pca_events, min_neurons=min_neurons, plot=False,
    )
    if no_PCs is None:
        no_PCs = pca_traj.PCs_needed(pc_result.explained_variance, percent_var / 100)
    num_pcs = no_PCs
    full_PCA_matrix = pc_result.matrix_df
    coefficients = pc_result.coefficients[:, :no_PCs]
    recordings = full_PCA_matrix.columns.to_list()
    recordings_arr = np.array(recordings)
    neuron_indices = {
        rec: np.where(recordings_arr == rec)[0]
        for rec in np.unique(recordings_arr)
    }
    scaler_mean = pc_result.scaler.mean_
    scaler_scale = pc_result.scaler.scale_
    W = pc_result.coefficients
    decoder_data = defaultdict(list)
    recording_labels = defaultdict(list)
    for recording in spike_collection.recordings:
        try:
            idx = neuron_indices[recording.name]
            subset_coeff = W[:no_PCs, idx].T        # [n_i, no_PCs]
            rec_mean = scaler_mean[idx]
            rec_scale = scaler_scale[idx]
            for event in project_events:
                event_firing_rates = recording.event_firing_rates(event, event_length, pre_window, post_window)
                for trial in range(len(event_firing_rates)):
                    normalized_trial = (event_firing_rates[trial] - rec_mean) / rec_scale
                    trial_data = np.dot(normalized_trial, subset_coeff)
                    decoder_data[event].append(trial_data)
                    recording_labels[event].append(recording.name)
        except KeyError:
            pass
    if return_recording_labels:
        return decoder_data, num_pcs, recording_labels
    return decoder_data, num_pcs


def _cross_gen_decoder(decoder_data_by_pair, events, num_fold, classifier_type, input="timebin", recording_labels_by_pair=None, **kwargs):
    """Core cross-generalization decoder.

    For each pairwise training set (e1, e2), trains a classifier per fold per timebin,
    then scores on two generalization test sets using the held-out event (e3).

    decoder_data_by_pair : {train_key: decoder_data}
        Each train_key (e.g. "exp1 sniff_exp2 sniff") maps to its own decoder_data dict,
        which may have been projected from a PCA fit on only those two events (hold_out=True)
        or from a shared PCA across all events (hold_out=False).

    Label rule:
      - e1=0, e2=1 in training
      - Test (e1 vs e3): e1=0, e3=1  (e3 gets e2's label)
      - Test (e2 vs e3): e2=1, e3=0  (e3 gets e1's label)

    input : str, 'timebin' or 'full_trial'
        'timebin'    — fit one classifier per timebin (original behaviour)
        'full_trial' — flatten each trial (T*n_PCs features) and fit one classifier per fold

    recording_labels_by_pair : optional {train_key: {event: [rec_name, ...]}}
        If provided, use leave-one-recording-out (LOO) CV instead of k-fold.
        num_fold is ignored; effective fold count = number of unique recordings.

    Returns:
        raw_results : {train_key: {test_key: list of dicts with 'test_roc_auc': array(num_fold,)}}
            For 'timebin' the list has T entries; for 'full_trial' it has 1 entry.
    """
    assert len(events) == 3, "cross-generalization requires exactly 3 events"

    raw_results = {}

    for e1, e2 in combinations(events, 2):
        e3 = next(e for e in events if e != e1 and e != e2)
        train_key = f"{e1}_{e2}"
        decoder_data = decoder_data_by_pair[train_key]
        T = decoder_data[events[0]][0].shape[0]

        # (test_event1, test_event2, label_for_event1, label_for_event2)
        test_pairs = [
            (e1, e2, 0, 1),  # in-distribution: held-out fold of the trained pair
            (e1, e3, 0, 1),  # e3 gets e2's label (1)
            (e2, e3, 1, 0),  # e3 gets e1's label (0)
        ]
        shuffle_key = f"{e1}_{e2}_shuffle"

        if recording_labels_by_pair is not None:
            # --- LOO path: leave one recording out per fold ---
            rec_labels = recording_labels_by_pair[train_key]
            all_recs = list(dict.fromkeys(r for ev in events for r in rec_labels[ev]))
            N = len(all_recs)

            if input == "full_trial":
                auc_arrays = {f"{te1}_{te2}": np.full((1, N), np.nan) for te1, te2, _, _ in test_pairs}
                auc_arrays[shuffle_key] = np.full((1, N), np.nan)
                fold_models = []

                for fold_idx, rec_name in enumerate(all_recs):
                    train_e1 = [tr for tr, r in zip(decoder_data[e1], rec_labels[e1]) if r != rec_name]
                    train_e2 = [tr for tr, r in zip(decoder_data[e2], rec_labels[e2]) if r != rec_name]
                    if not train_e1 or not train_e2:
                        continue
                    X_train_all = np.concatenate(
                        [np.stack(train_e1, axis=2), np.stack(train_e2, axis=2)], axis=2
                    ).transpose()
                    y_train = np.array([0] * len(train_e1) + [1] * len(train_e2))
                    X_train_flat = X_train_all.reshape(X_train_all.shape[0], -1)
                    clf = __fit_clf_single__(X_train_flat, y_train, classifier_type, **kwargs)
                    clf_shuf = __fit_clf_single__(X_train_flat, np.random.permutation(y_train), classifier_type, **kwargs)
                    fold_models.append(clf)

                    X_test_flat_by_pair = {}
                    y_test_by_pair = {}
                    for te1, te2, lab1, lab2 in test_pairs:
                        tkey = f"{te1}_{te2}"
                        fold_te1 = [tr for tr, r in zip(decoder_data[te1], rec_labels[te1]) if r == rec_name]
                        fold_te2 = [tr for tr, r in zip(decoder_data[te2], rec_labels[te2]) if r == rec_name]
                        if fold_te1 and fold_te2:
                            X_t = np.concatenate(
                                [np.stack(fold_te1, axis=2), np.stack(fold_te2, axis=2)], axis=2
                            ).transpose()
                            X_test_flat_by_pair[tkey] = X_t.reshape(X_t.shape[0], -1)
                            y_test_by_pair[tkey] = np.array([lab1] * len(fold_te1) + [lab2] * len(fold_te2))

                    for tkey, X_test_flat in X_test_flat_by_pair.items():
                        y_t = y_test_by_pair[tkey]
                        if len(np.unique(y_t)) >= 2:
                            auc_arrays[tkey][0, fold_idx] = __score_test__(clf, X_test_flat, y_t, classifier_type)

                    in_dist_key = f"{e1}_{e2}"
                    if in_dist_key in X_test_flat_by_pair:
                        y_t = y_test_by_pair[in_dist_key]
                        if len(np.unique(y_t)) >= 2:
                            auc_arrays[shuffle_key][0, fold_idx] = __score_test__(clf_shuf, X_test_flat_by_pair[in_dist_key], y_t, classifier_type)

                stored_models = [fold_models]

            else:
                auc_arrays = {f"{te1}_{te2}": np.full((T, N), np.nan) for te1, te2, _, _ in test_pairs}
                auc_arrays[shuffle_key] = np.full((T, N), np.nan)
                fold_models_by_time = [[] for _ in range(T)]

                for fold_idx, rec_name in enumerate(all_recs):
                    train_e1 = [tr for tr, r in zip(decoder_data[e1], rec_labels[e1]) if r != rec_name]
                    train_e2 = [tr for tr, r in zip(decoder_data[e2], rec_labels[e2]) if r != rec_name]
                    if not train_e1 or not train_e2:
                        continue
                    X_train_all = np.concatenate(
                        [np.stack(train_e1, axis=2), np.stack(train_e2, axis=2)], axis=2
                    ).transpose()
                    y_train = np.array([0] * len(train_e1) + [1] * len(train_e2))

                    X_test_by_pair = {}
                    y_test_by_pair = {}
                    for te1, te2, lab1, lab2 in test_pairs:
                        tkey = f"{te1}_{te2}"
                        fold_te1 = [tr for tr, r in zip(decoder_data[te1], rec_labels[te1]) if r == rec_name]
                        fold_te2 = [tr for tr, r in zip(decoder_data[te2], rec_labels[te2]) if r == rec_name]
                        if fold_te1 and fold_te2:
                            X_test_by_pair[tkey] = np.concatenate(
                                [np.stack(fold_te1, axis=2), np.stack(fold_te2, axis=2)], axis=2
                            ).transpose()
                            y_test_by_pair[tkey] = np.array([lab1] * len(fold_te1) + [lab2] * len(fold_te2))

                    for t in range(T):
                        clf = __fit_clf_single__(X_train_all[:, :, t], y_train, classifier_type, **kwargs)
                        fold_models_by_time[t].append(clf)
                        for tkey, X_test in X_test_by_pair.items():
                            y_t = y_test_by_pair[tkey]
                            if len(np.unique(y_t)) >= 2:
                                auc_arrays[tkey][t, fold_idx] = __score_test__(clf, X_test[:, :, t], y_t, classifier_type)
                        clf_shuf = __fit_clf_single__(
                            X_train_all[:, :, t], np.random.permutation(y_train), classifier_type, **kwargs
                        )
                        in_dist_key = f"{e1}_{e2}"
                        if in_dist_key in X_test_by_pair and len(np.unique(y_test_by_pair[in_dist_key])) >= 2:
                            auc_arrays[shuffle_key][t, fold_idx] = __score_test__(
                                clf_shuf, X_test_by_pair[in_dist_key][:, :, t], y_test_by_pair[in_dist_key], classifier_type
                            )

                stored_models = fold_models_by_time

        else:
            # --- original k-fold path ---
            event_folds = {event: __split_into_folds__(decoder_data[event], num_fold) for event in events}

            if input == "full_trial":
                auc_arrays = {f"{te1}_{te2}": np.full((1, num_fold), np.nan) for te1, te2, _, _ in test_pairs}
                auc_arrays[shuffle_key] = np.full((1, num_fold), np.nan)
                fold_models = []

                for k in range(num_fold):
                    X_train_all, y_train = __build_train_fold__(k, event_folds, e1, e2)
                    X_test_per_pair, y_test_per_pair = __build_test_fold__(k, event_folds, test_pairs)

                    X_train_flat = X_train_all.reshape(X_train_all.shape[0], -1)
                    clf = __fit_clf_single__(X_train_flat, y_train, classifier_type, **kwargs)
                    fold_models.append(clf)

                    for te1, te2, _, _ in test_pairs:
                        test_key = f"{te1}_{te2}"
                        X_test_flat = X_test_per_pair[test_key].reshape(X_test_per_pair[test_key].shape[0], -1)
                        auc_arrays[test_key][0, k] = __score_test__(clf, X_test_flat, y_test_per_pair[test_key], classifier_type)

                    clf_shuf = __fit_clf_single__(X_train_flat, np.random.permutation(y_train), classifier_type, **kwargs)
                    in_dist_key = f"{e1}_{e2}"
                    X_test_flat = X_test_per_pair[in_dist_key].reshape(X_test_per_pair[in_dist_key].shape[0], -1)
                    auc_arrays[shuffle_key][0, k] = __score_test__(clf_shuf, X_test_flat, y_test_per_pair[in_dist_key], classifier_type)

                stored_models = [fold_models]

            else:
                auc_arrays = {f"{te1}_{te2}": np.full((T, num_fold), np.nan) for te1, te2, _, _ in test_pairs}
                auc_arrays[shuffle_key] = np.full((T, num_fold), np.nan)
                fold_models_by_time = [[] for _ in range(T)]

                for k in range(num_fold):
                    X_train_all, y_train = __build_train_fold__(k, event_folds, e1, e2)
                    X_test_per_pair, y_test_per_pair = __build_test_fold__(k, event_folds, test_pairs)

                    for t in range(T):
                        clf = __fit_clf_single__(X_train_all[:, :, t], y_train, classifier_type, **kwargs)
                        fold_models_by_time[t].append(clf)
                        for te1, te2, _, _ in test_pairs:
                            test_key = f"{te1}_{te2}"
                            auc_arrays[test_key][t, k] = __score_test__(
                                clf, X_test_per_pair[test_key][:, :, t], y_test_per_pair[test_key], classifier_type
                            )
                        clf_shuf = __fit_clf_single__(
                            X_train_all[:, :, t], np.random.permutation(y_train), classifier_type, **kwargs
                        )
                        in_dist_key = f"{e1}_{e2}"
                        auc_arrays[shuffle_key][t, k] = __score_test__(
                            clf_shuf, X_test_per_pair[in_dist_key][:, :, t], y_test_per_pair[in_dist_key], classifier_type
                        )

                stored_models = fold_models_by_time

        # convert to list-of-dicts format (T entries for timebin, 1 entry for full_trial)
        raw_results[train_key] = {"_models": stored_models}
        for test_key, auc_arr in auc_arrays.items():
            raw_results[train_key][test_key] = [
                {"test_roc_auc": auc_arr[t, :]} for t in range(auc_arr.shape[0])
            ]

    return raw_results


def trial_decoder_cross_generalization(
    spike_collection,
    num_fold,
    events,
    event_length,
    percent_var=90,
    pre_window=0,
    post_window=0,
    min_neurons=0,
    condition_dict=None,
    classifier_type="RF",
    hold_out=True,
    LOO=False,
    input="timebin",
    no_PCs=None,
    **kwargs,
):
    """Cross-generalization decoder for exactly 3 events.

    Trains on each pairwise combination and tests generalization to the held-out event.
    See cross_gen_results for output structure.

    hold_out : bool, default True
        If True, fit a separate PCA for each training pair (e1, e2) using only those two
        events' averaged trajectories. PCs are chosen to explain percent_var% of that
        pair's variance — so each pair may have a different num_pcs.
        If False, fit one shared PCA across all three events; num_pcs is the same for all pairs.
    LOO : bool, default False
        If True, use leave-one-recording-out CV for the classifier instead of k-fold.
        num_fold is ignored; effective fold count = number of unique recordings.
        The PCA space is still fit on all recordings (required by the pooled-neuron architecture).
    """
    assert len(events) == 3, "cross-generalization requires exactly 3 events"
    recording_labels_by_pair = None
    if LOO:
        if hold_out:
            decoder_data_by_pair = {}
            num_pcs_by_pair = {}
            recording_labels_by_pair = {}
            for e1, e2 in combinations(events, 2):
                train_key = f"{e1}_{e2}"
                decoder_data_by_pair[train_key], num_pcs_by_pair[train_key], recording_labels_by_pair[train_key] = _trial_PCA_holdout(
                    spike_collection, event_length, pre_window, post_window,
                    pca_events=[e1, e2], project_events=events,
                    percent_var=percent_var, min_neurons=min_neurons,
                    condition_dict=condition_dict, no_PCs=no_PCs,
                    return_recording_labels=True,
                )
        else:
            single_data, num_pcs, shared_labels = trial_PCA(
                spike_collection, event_length, pre_window, post_window,
                percent_var=percent_var, events=events, min_neurons=min_neurons,
                condition_dict=condition_dict, no_PCs=no_PCs,
                return_recording_labels=True,
            )
            decoder_data_by_pair = {f"{e1}_{e2}": single_data for e1, e2 in combinations(events, 2)}
            num_pcs_by_pair = {f"{e1}_{e2}": num_pcs for e1, e2 in combinations(events, 2)}
            recording_labels_by_pair = {f"{e1}_{e2}": shared_labels for e1, e2 in combinations(events, 2)}
        first_labels = next(iter(recording_labels_by_pair.values()))
        all_recs = list(dict.fromkeys(r for ev in events for r in first_labels[ev]))
        effective_num_fold = len(all_recs)
    else:
        if hold_out:
            decoder_data_by_pair = {}
            num_pcs_by_pair = {}
            for e1, e2 in combinations(events, 2):
                train_key = f"{e1}_{e2}"
                decoder_data_by_pair[train_key], num_pcs_by_pair[train_key] = _trial_PCA_holdout(
                    spike_collection, event_length, pre_window, post_window,
                    pca_events=[e1, e2], project_events=events,
                    percent_var=percent_var, min_neurons=min_neurons,
                    condition_dict=condition_dict, no_PCs=no_PCs,
                )
        else:
            single_data, num_pcs = trial_PCA(
                spike_collection, event_length, pre_window, post_window,
                percent_var=percent_var, events=events, min_neurons=min_neurons,
                condition_dict=condition_dict, no_PCs=no_PCs,
            )
            decoder_data_by_pair = {f"{e1}_{e2}": single_data for e1, e2 in combinations(events, 2)}
            num_pcs_by_pair = {f"{e1}_{e2}": num_pcs for e1, e2 in combinations(events, 2)}
        effective_num_fold = num_fold
    raw_results = _cross_gen_decoder(
        decoder_data_by_pair, events, num_fold, classifier_type, input=input,
        recording_labels_by_pair=recording_labels_by_pair, **kwargs,
    )
    return cross_gen_results(raw_results, effective_num_fold, event_length, pre_window, post_window, percent_var=percent_var, num_pcs_by_pair=num_pcs_by_pair)


class nested_model_result:
    """Parallel to model_results — holds roc_auc (T, num_fold) for one test pair."""

    def __init__(self, timebin_list, num_fold):
        T = len(timebin_list)
        self.roc_auc = np.empty((T, num_fold))
        for t, d in enumerate(timebin_list):
            self.roc_auc[t] = d["test_roc_auc"]
        self.avg_auc = np.nanmean(self.roc_auc)

    def __repr__(self):
        return f"nested_model_result | avg AUC: {self.avg_auc:.3f}"


class cross_gen_results:
    """Parallel to all_results — results from trial_decoder_cross_generalization.

    Attributes
    ----------
    roc_auc_scores : dict
        {train_pair_key: {test_pair_key: nested_model_result}}
        e.g. {"A_B": {"A_C": nested_model_result, "B_C": nested_model_result}, ...}
    """

    def __init__(self, raw_results, num_fold, event_length, pre_window, post_window, percent_var=None, num_pcs_by_pair=None):
        self.num_fold = num_fold
        self.event_length = event_length
        self.pre_window = pre_window
        self.post_window = post_window
        self.percent_var = percent_var
        self.num_pcs_by_pair = num_pcs_by_pair or {}
        self.models = {}
        self.roc_auc_scores = {}
        for train_key, test_pairs in raw_results.items():
            self.models[train_key] = test_pairs.pop("_models", None)
            self.roc_auc_scores[train_key] = {
                test_key: nested_model_result(timebin_list, num_fold)
                for test_key, timebin_list in test_pairs.items()
            }

    def __repr__(self):
        lines = [f"Cross-generalization decoder | {self.num_fold} folds"]
        if self.percent_var is not None:
            lines.append(f"Variance threshold: {self.percent_var}%")
        for train_key, test_pairs in self.roc_auc_scores.items():
            n = self.num_pcs_by_pair.get(train_key)
            pcs_str = f" [{n} PCs]" if n is not None else ""
            lines.append(f"  Trained on {train_key}{pcs_str}:")
            for test_key, res in test_pairs.items():
                lines.append(f"    → tested on {test_key}: avg AUC = {res.avg_auc:.3f}")
        return "\n".join(lines)

    def plot_across_time(self, start=None, stop=None):
        """One subplot per test case (6 total for 3 events), mean ± SEM across folds."""
        all_test_cases = [
            (train_key, test_key, res)
            for train_key, test_pairs in self.roc_auc_scores.items()
            for test_key, res in test_pairs.items()
        ]
        n = len(all_test_cases)
        ncols = 2
        nrows = math.ceil(n / ncols)
        if start is None:
            start = -self.pre_window
        if stop is None:
            stop = self.event_length + self.post_window
        plt.figure(figsize=(12, 4 * nrows))
        for i, (train_key, test_key, res) in enumerate(all_test_cases, 1):
            plt.subplot(nrows, ncols, i)
            x = np.linspace(-self.pre_window, self.event_length + self.post_window, res.roc_auc.shape[0])
            avg = np.nanmean(res.roc_auc, axis=1)
            err = sem(res.roc_auc, axis=1, nan_policy="omit")
            plt.plot(x, avg)
            plt.fill_between(x, avg - err, avg + err, alpha=0.2)
            plt.axhline(0.5, color="k", linestyle="--", linewidth=0.8)
            plt.axvline(0, color="k", linestyle="--", linewidth=0.8)
            plt.ylim(0.3, 1.0)
            plt.title(f"Train: {train_key}  →  Test: {test_key}")
            plt.ylabel("ROC AUC")
        plt.suptitle("Cross-generalization decoder")
        plt.tight_layout()
        plt.show()

    def plot_average(self, start=0, stop=None):
        """Bar plot of ROC AUC averaged over a time window.

        One subplot per training pair (3 total), three bars each — the trained pair
        (in-distribution) and the two generalization test cases. Mean and SEM are
        computed by first averaging over timebins in [start, stop], then taking
        mean ± SEM across folds.

        Args:
            start : float, seconds from event onset to start averaging (default 0)
            stop  : float, seconds from event onset to stop averaging (default event_length)
        """
        train_keys = list(self.roc_auc_scores.keys())
        plt.figure(figsize=(5 * len(train_keys), 4))
        bar_width = 0.5

        for i, train_key in enumerate(train_keys, 1):
            plt.subplot(1, len(train_keys), i)
            test_pairs = self.roc_auc_scores[train_key]
            test_keys = list(test_pairs.keys())

            shuffle_key = f"{train_key}_shuffle"
            display_keys = [k for k in test_keys if k != shuffle_key]
            bar_labels = display_keys + (["shuffle"] if shuffle_key in test_pairs else [])
            ordered_keys = display_keys + ([shuffle_key] if shuffle_key in test_pairs else [])

            means, errors = [], []
            for test_key in ordered_keys:
                nmr = test_pairs[test_key]
                x = np.linspace(
                    -self.pre_window,
                    self.event_length + self.post_window,
                    nmr.roc_auc.shape[0],
                )
                plot_start = np.where(x >= start)[0][0] if np.any(x >= start) else 0
                if stop is None:
                    plot_stop = nmr.roc_auc.shape[0]
                else:
                    plot_stop = np.where(x <= stop)[0][-1] + 1

                # average over timebins first → (num_fold,), then mean ± SEM across folds
                avg_per_fold = np.nanmean(nmr.roc_auc[plot_start:plot_stop], axis=0)
                means.append(np.nanmean(avg_per_fold))
                errors.append(sem(avg_per_fold, nan_policy="omit"))

            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            bar_colors = [colors[j % len(colors)] for j in range(len(display_keys))] + (
                ["lightgray"] if shuffle_key in test_pairs else []
            )
            x_pos = np.arange(len(ordered_keys))
            plt.bar(x_pos, means, bar_width, yerr=errors, capsize=5, color=bar_colors)
            plt.xticks(x_pos, bar_labels, rotation=15, ha="right")
            plt.axhline(0.5, color="k", linestyle="--", linewidth=0.8)
            plt.ylim(0.3, 1.0)
            plt.ylabel("ROC AUC")
            plt.title(f"Train: {train_key}")

        plt.suptitle("Cross-generalization decoder (time-averaged)")
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# 2x2 Cross-generalization decoder
# ---------------------------------------------------------------------------

def _cross_gen_decoder_2x2(decoder_data_by_pair, train_test_pairs, num_fold, classifier_type, input="timebin", recording_labels_by_pair=None, n_shuffles=5, **kwargs):
    """Core 2x2 cross-generalization decoder.

    For each (train_pair, test_pair) direction, runs num_fold folds producing:
      - in_dist  : train on (e1, e2), test held-out fold of (e1, e2)
      - cross_gen: same model, test fold of (te1, te2) with te1=0, te2=1
      - shuffle  : train on shuffled labels, test held-out fold of (e1, e2)

    Args:
        decoder_data_by_pair : {train_key: decoder_data}
            Each decoder_data must contain all 4 events projected into that pair's PCA space.
        train_test_pairs : list of ((e1, e2), (te1, te2)) tuples
        num_fold         : int
        classifier_type  : "RF" or "linear"
        input            : "timebin" or "full_trial"
        recording_labels_by_pair : optional {train_key: {event: [rec_name, ...]}}
            If provided, use leave-one-recording-out CV instead of k-fold.

    Returns:
        raw_results : {train_key: {"in_dist": [...], "cross_gen": [...], "shuffle": [...]}}
            Each list has T entries (timebin) or 1 entry (full_trial),
            each entry is {"test_roc_auc": np.ndarray(num_fold,)}.
    """
    raw_results = {}

    for (e1, e2), (te1, te2) in train_test_pairs:
        train_key = f"{e1}_{e2}"
        decoder_data = decoder_data_by_pair[train_key]
        all_events = list(dict.fromkeys([e1, e2, te1, te2]))
        T_dim = 1 if input == "full_trial" else decoder_data[e1][0].shape[0]

        if recording_labels_by_pair is not None:
            # --- LOO path ---
            rec_labels = recording_labels_by_pair[train_key]
            all_recs = list(dict.fromkeys(r for ev in all_events for r in rec_labels[ev]))
            N = len(all_recs)

            auc_in_dist   = np.full((T_dim, N), np.nan)
            auc_cross_gen = np.full((T_dim, N), np.nan)
            auc_shuffle   = np.full((T_dim, N), np.nan)

            if input == "full_trial":
                fold_models = []
                for fold_idx, rec_name in enumerate(all_recs):
                    train_e1 = [tr for tr, r in zip(decoder_data[e1], rec_labels[e1]) if r != rec_name]
                    train_e2 = [tr for tr, r in zip(decoder_data[e2], rec_labels[e2]) if r != rec_name]
                    if not train_e1 or not train_e2:
                        continue
                    X_train_all = np.concatenate(
                        [np.stack(train_e1, axis=2), np.stack(train_e2, axis=2)], axis=2
                    ).transpose()
                    y_train = np.array([0] * len(train_e1) + [1] * len(train_e2))
                    X_train_flat = X_train_all.reshape(X_train_all.shape[0], -1)
                    clf = __fit_clf_single__(X_train_flat, y_train, classifier_type, **kwargs)
                    fold_models.append(clf)

                    in_e1  = [tr for tr, r in zip(decoder_data[e1],  rec_labels[e1])  if r == rec_name]
                    in_e2  = [tr for tr, r in zip(decoder_data[e2],  rec_labels[e2])  if r == rec_name]
                    cr_te1 = [tr for tr, r in zip(decoder_data[te1], rec_labels[te1]) if r == rec_name]
                    cr_te2 = [tr for tr, r in zip(decoder_data[te2], rec_labels[te2]) if r == rec_name]

                    if in_e1 and in_e2:
                        X_in = np.concatenate([np.stack(in_e1, axis=2), np.stack(in_e2, axis=2)], axis=2).transpose()
                        y_in = np.array([0] * len(in_e1) + [1] * len(in_e2))
                        if len(np.unique(y_in)) >= 2:
                            X_in_flat = X_in.reshape(X_in.shape[0], -1)
                            auc_in_dist[0, fold_idx]  = __score_test__(clf, X_in_flat, y_in, classifier_type)
                            auc_shuffle[0, fold_idx]  = np.mean([
                                __score_test__(__fit_clf_single__(X_train_flat, np.random.permutation(y_train), classifier_type, **kwargs), X_in_flat, y_in, classifier_type)
                                for _ in range(n_shuffles)
                            ])
                    if cr_te1 and cr_te2:
                        X_cross = np.concatenate([np.stack(cr_te1, axis=2), np.stack(cr_te2, axis=2)], axis=2).transpose()
                        y_cross = np.array([0] * len(cr_te1) + [1] * len(cr_te2))
                        if len(np.unique(y_cross)) >= 2:
                            X_cross_flat = X_cross.reshape(X_cross.shape[0], -1)
                            auc_cross_gen[0, fold_idx] = __score_test__(clf, X_cross_flat, y_cross, classifier_type)
                stored_models = [fold_models]
            else:
                fold_models_by_time = [[] for _ in range(T_dim)]
                for fold_idx, rec_name in enumerate(all_recs):
                    train_e1 = [tr for tr, r in zip(decoder_data[e1], rec_labels[e1]) if r != rec_name]
                    train_e2 = [tr for tr, r in zip(decoder_data[e2], rec_labels[e2]) if r != rec_name]
                    if not train_e1 or not train_e2:
                        continue
                    X_train_all = np.concatenate(
                        [np.stack(train_e1, axis=2), np.stack(train_e2, axis=2)], axis=2
                    ).transpose()
                    y_train = np.array([0] * len(train_e1) + [1] * len(train_e2))

                    in_e1  = [tr for tr, r in zip(decoder_data[e1],  rec_labels[e1])  if r == rec_name]
                    in_e2  = [tr for tr, r in zip(decoder_data[e2],  rec_labels[e2])  if r == rec_name]
                    cr_te1 = [tr for tr, r in zip(decoder_data[te1], rec_labels[te1]) if r == rec_name]
                    cr_te2 = [tr for tr, r in zip(decoder_data[te2], rec_labels[te2]) if r == rec_name]

                    has_in    = bool(in_e1 and in_e2)
                    has_cross = bool(cr_te1 and cr_te2)
                    if has_in:
                        X_in_all = np.concatenate([np.stack(in_e1, axis=2), np.stack(in_e2, axis=2)], axis=2).transpose()
                        y_in = np.array([0] * len(in_e1) + [1] * len(in_e2))
                    if has_cross:
                        X_cross_all = np.concatenate([np.stack(cr_te1, axis=2), np.stack(cr_te2, axis=2)], axis=2).transpose()
                        y_cross = np.array([0] * len(cr_te1) + [1] * len(cr_te2))

                    for t in range(T_dim):
                        clf = __fit_clf_single__(X_train_all[:, :, t], y_train, classifier_type, **kwargs)
                        fold_models_by_time[t].append(clf)
                        if has_in and len(np.unique(y_in)) >= 2:
                            auc_in_dist[t, fold_idx]  = __score_test__(clf, X_in_all[:, :, t], y_in, classifier_type)
                            auc_shuffle[t, fold_idx]  = np.mean([
                                __score_test__(__fit_clf_single__(X_train_all[:, :, t], np.random.permutation(y_train), classifier_type, **kwargs), X_in_all[:, :, t], y_in, classifier_type)
                                for _ in range(n_shuffles)
                            ])
                        if has_cross and len(np.unique(y_cross)) >= 2:
                            auc_cross_gen[t, fold_idx] = __score_test__(clf, X_cross_all[:, :, t], y_cross, classifier_type)
                stored_models = fold_models_by_time

        else:
            # --- original k-fold path ---
            event_folds = {event: __split_into_folds__(decoder_data[event], num_fold) for event in all_events}

            auc_in_dist   = np.full((T_dim, num_fold), np.nan)
            auc_cross_gen = np.full((T_dim, num_fold), np.nan)
            auc_shuffle   = np.full((T_dim, num_fold), np.nan)

            in_dist_pairs   = [(e1,  e2,  0, 1)]
            cross_gen_pairs = [(te1, te2, 0, 1)]
            in_key    = f"{e1}_{e2}"
            cross_key = f"{te1}_{te2}"

            if input == "full_trial":
                fold_models = []
            else:
                fold_models_by_time = [[] for _ in range(T_dim)]

            for k in range(num_fold):
                X_train_all, y_train = __build_train_fold__(k, event_folds, e1, e2)
                X_in,    y_in    = __build_test_fold__(k, event_folds, in_dist_pairs)
                X_cross, y_cross = __build_test_fold__(k, event_folds, cross_gen_pairs)

                if input == "full_trial":
                    X_train_flat  = X_train_all.reshape(X_train_all.shape[0], -1)
                    X_in_flat     = X_in[in_key].reshape(X_in[in_key].shape[0], -1)
                    X_cross_flat  = X_cross[cross_key].reshape(X_cross[cross_key].shape[0], -1)

                    clf = __fit_clf_single__(X_train_flat, y_train, classifier_type, **kwargs)
                    fold_models.append(clf)

                    auc_in_dist[0, k]   = __score_test__(clf, X_in_flat,    y_in[in_key],       classifier_type)
                    auc_cross_gen[0, k] = __score_test__(clf, X_cross_flat, y_cross[cross_key], classifier_type)
                    auc_shuffle[0, k]   = np.mean([
                        __score_test__(__fit_clf_single__(X_train_flat, np.random.permutation(y_train), classifier_type, **kwargs), X_in_flat, y_in[in_key], classifier_type)
                        for _ in range(n_shuffles)
                    ])
                else:
                    for t in range(T_dim):
                        clf = __fit_clf_single__(X_train_all[:, :, t], y_train, classifier_type, **kwargs)
                        fold_models_by_time[t].append(clf)

                        auc_in_dist[t, k]   = __score_test__(clf, X_in[in_key][:, :, t],       y_in[in_key],       classifier_type)
                        auc_cross_gen[t, k] = __score_test__(clf, X_cross[cross_key][:, :, t], y_cross[cross_key], classifier_type)
                        auc_shuffle[t, k]   = np.mean([
                            __score_test__(__fit_clf_single__(X_train_all[:, :, t], np.random.permutation(y_train), classifier_type, **kwargs), X_in[in_key][:, :, t], y_in[in_key], classifier_type)
                            for _ in range(n_shuffles)
                        ])

            stored_models = [fold_models] if input == "full_trial" else fold_models_by_time

        raw_results[train_key] = {
            "_models":   stored_models,
            "in_dist":   [{"test_roc_auc": auc_in_dist[t, :]}   for t in range(T_dim)],
            "cross_gen": [{"test_roc_auc": auc_cross_gen[t, :]} for t in range(T_dim)],
            "shuffle":   [{"test_roc_auc": auc_shuffle[t, :]}   for t in range(T_dim)],
        }

    return raw_results


def trial_decoder_cross_generalization_2x2(
    spike_collection,
    num_fold,
    events,
    event_length,
    percent_var=90,
    pre_window=0,
    post_window=0,
    min_neurons=0,
    condition_dict=None,
    classifier_type="RF",
    hold_out=True,
    LOO=False,
    input="timebin",
    no_PCs=None,
    n_shuffles=5,
    **kwargs,
):
    """2x2 cross-generalization decoder.

    Trains on each axis of a 2x2 event structure and tests generalization across the other axis.

    Args:
        spike_collection : SpikeCollection
        num_fold         : int, number of CV folds
        events           : [[a, b], [c, d]]
            Row axis: a vs b  ↔  c vs d
            Col axis: a vs c  ↔  b vs d
        event_length     : float, seconds
        hold_out         : bool, default True
            If True, fit a separate PCA per training pair (same as trial_decoder_cross_generalization).
            If False, one shared PCA across all 4 events.
        LOO : bool, default False
            If True, use leave-one-recording-out CV for the classifier instead of k-fold.
            num_fold is ignored; effective fold count = number of unique recordings.
        input            : "timebin" or "full_trial"

    Returns:
        cross_gen_2x2_results
    """
    a, b = events[0]
    c, d = events[1]
    all_events = [a, b, c, d]

    train_test_pairs = [
        ((a, b), (c, d)),  # axis 1, direction 1
        ((c, d), (a, b)),  # axis 1, direction 2
        ((a, c), (b, d)),  # axis 2, direction 1
        ((b, d), (a, c)),  # axis 2, direction 2
        ((a, d), (b, c)),  # xor, direction 1
        ((b, c), (a, d)),  # xor, direction 2
    ]

    recording_labels_by_pair = None
    if LOO:
        if hold_out:
            decoder_data_by_pair = {}
            num_pcs_by_pair = {}
            recording_labels_by_pair = {}
            for (e1, e2), _ in train_test_pairs:
                train_key = f"{e1}_{e2}"
                decoder_data_by_pair[train_key], num_pcs_by_pair[train_key], recording_labels_by_pair[train_key] = _trial_PCA_holdout(
                    spike_collection, event_length, pre_window, post_window,
                    pca_events=[e1, e2], project_events=all_events,
                    percent_var=percent_var, min_neurons=min_neurons,
                    condition_dict=condition_dict, no_PCs=no_PCs,
                    return_recording_labels=True,
                )
        else:
            single_data, num_pcs, shared_labels = trial_PCA(
                spike_collection, event_length, pre_window, post_window,
                percent_var=percent_var, events=all_events, min_neurons=min_neurons,
                condition_dict=condition_dict, no_PCs=no_PCs,
                return_recording_labels=True,
            )
            decoder_data_by_pair = {f"{e1}_{e2}": single_data for (e1, e2), _ in train_test_pairs}
            num_pcs_by_pair = {f"{e1}_{e2}": num_pcs for (e1, e2), _ in train_test_pairs}
            recording_labels_by_pair = {f"{e1}_{e2}": shared_labels for (e1, e2), _ in train_test_pairs}
        first_labels = next(iter(recording_labels_by_pair.values()))
        all_recs = list(dict.fromkeys(r for ev in first_labels for r in first_labels[ev]))
        effective_num_fold = len(all_recs)
    else:
        if hold_out:
            decoder_data_by_pair = {}
            num_pcs_by_pair = {}
            for (e1, e2), _ in train_test_pairs:
                train_key = f"{e1}_{e2}"
                decoder_data_by_pair[train_key], num_pcs_by_pair[train_key] = _trial_PCA_holdout(
                    spike_collection, event_length, pre_window, post_window,
                    pca_events=[e1, e2], project_events=all_events,
                    percent_var=percent_var, min_neurons=min_neurons,
                    condition_dict=condition_dict, no_PCs=no_PCs,
                )
        else:
            single_data, num_pcs = trial_PCA(
                spike_collection, event_length, pre_window, post_window,
                percent_var=percent_var, events=all_events, min_neurons=min_neurons,
                condition_dict=condition_dict, no_PCs=no_PCs,
            )
            decoder_data_by_pair = {f"{e1}_{e2}": single_data for (e1, e2), _ in train_test_pairs}
            num_pcs_by_pair = {f"{e1}_{e2}": num_pcs for (e1, e2), _ in train_test_pairs}
        effective_num_fold = num_fold

    raw_results = _cross_gen_decoder_2x2(
        decoder_data_by_pair, train_test_pairs, num_fold, classifier_type, input=input,
        recording_labels_by_pair=recording_labels_by_pair, n_shuffles=n_shuffles, **kwargs,
    )
    return cross_gen_2x2_results(
        raw_results, train_test_pairs, effective_num_fold, event_length, pre_window, post_window,
        percent_var=percent_var, num_pcs_by_pair=num_pcs_by_pair,
    )


class cross_gen_2x2_results:
    """Results from trial_decoder_cross_generalization_2x2.

    Attributes
    ----------
    roc_auc_scores : dict
        {train_key: {"in_dist": nested_model_result,
                     "cross_gen": nested_model_result,
                     "shuffle": nested_model_result}}
    train_test_pairs : list of ((e1, e2), (te1, te2))
    axis_map : dict {train_key: "axis1" or "axis2"}
    """

    def __init__(self, raw_results, train_test_pairs, num_fold, event_length, pre_window, post_window,
                 percent_var=None, num_pcs_by_pair=None):
        self.num_fold = num_fold
        self.event_length = event_length
        self.pre_window = pre_window
        self.post_window = post_window
        self.percent_var = percent_var
        self.num_pcs_by_pair = num_pcs_by_pair or {}
        self.train_test_pairs = train_test_pairs
        self.axis_map = {
            f"{e1}_{e2}": ("axis1" if i < 2 else "axis2" if i < 4 else "xor")
            for i, ((e1, e2), _) in enumerate(train_test_pairs)
        }
        self.models = {}
        self.roc_auc_scores = {}
        for (e1, e2), _ in train_test_pairs:
            train_key = f"{e1}_{e2}"
            pair_data = raw_results[train_key]
            self.models[train_key] = pair_data.pop("_models", None)
            self.roc_auc_scores[train_key] = {
                result_type: nested_model_result(timebin_list, num_fold)
                for result_type, timebin_list in pair_data.items()
            }

    def __repr__(self):
        lines = [f"2x2 Cross-generalization decoder | {self.num_fold} folds"]
        if self.percent_var is not None:
            lines.append(f"Variance threshold: {self.percent_var}%")
        for (e1, e2), (te1, te2) in self.train_test_pairs:
            train_key = f"{e1}_{e2}"
            n = self.num_pcs_by_pair.get(train_key)
            pcs_str = f" [{n} PCs]" if n is not None else ""
            lines.append(f"  [{self.axis_map[train_key]}] Train: {e1} vs {e2}{pcs_str}  →  Test: {te1} vs {te2}")
            for result_type, nmr in self.roc_auc_scores[train_key].items():
                lines.append(f"    {result_type}: avg AUC = {nmr.avg_auc:.3f}")
        return "\n".join(lines)

    def plot_across_time(self, start=None, stop=None):
        """4 subplots (one per direction), each showing in_dist, cross_gen, and shuffle."""
        n = len(self.train_test_pairs)
        ncols = 2
        nrows = math.ceil(n / ncols)
        if start is None:
            start = -self.pre_window
        if stop is None:
            stop = self.event_length + self.post_window
        plt.figure(figsize=(12, 4 * nrows))
        for i, ((e1, e2), (te1, te2)) in enumerate(self.train_test_pairs, 1):
            train_key = f"{e1}_{e2}"
            results = self.roc_auc_scores[train_key]
            T = results["in_dist"].roc_auc.shape[0]
            x = np.linspace(-self.pre_window, self.event_length + self.post_window, T)
            plt.subplot(nrows, ncols, i)
            for result_type, nmr in results.items():
                avg = np.nanmean(nmr.roc_auc, axis=1)
                err = sem(nmr.roc_auc, axis=1, nan_policy="omit")
                ls = "--" if result_type == "shuffle" else "-"
                plt.plot(x, avg, linestyle=ls, label=result_type)
                plt.fill_between(x, avg - err, avg + err, alpha=0.2)
            plt.axhline(0.5, color="k", linestyle="--", linewidth=0.8)
            plt.axvline(0, color="k", linestyle="--", linewidth=0.8)
            plt.ylim(0.3, 1.0)
            plt.title(f"[{self.axis_map[train_key]}] Train: {e1} vs {e2}  →  Test: {te1} vs {te2}")
            plt.ylabel("ROC AUC")
            if i == 2:
                plt.legend(bbox_to_anchor=(1, 1))
        plt.suptitle("2x2 Cross-generalization decoder")
        plt.tight_layout()
        plt.show()

    def plot_average(self, start=0, stop=None):
        """Bar plot of epoch-averaged AUC, one subplot per direction."""
        n = len(self.train_test_pairs)
        fig, axes = plt.subplots(3, 2, figsize=(8, 12))
        axes = axes.flatten()
        bar_width = 0.4
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for ax, ((e1, e2), (te1, te2)) in zip(axes, self.train_test_pairs):
            train_key = f"{e1}_{e2}"
            results = self.roc_auc_scores[train_key]
            means, errors, labels, bar_colors = [], [], [], []
            for j, (result_type, nmr) in enumerate(results.items()):
                T = nmr.roc_auc.shape[0]
                x = np.linspace(-self.pre_window, self.event_length + self.post_window, T)
                start_idx = np.searchsorted(x, start)
                stop_idx  = T if stop is None else np.searchsorted(x, stop)
                avg_per_fold = np.nanmean(nmr.roc_auc[start_idx:stop_idx], axis=0)
                means.append(np.nanmean(avg_per_fold))
                errors.append(sem(avg_per_fold, nan_policy="omit"))
                labels.append(result_type)
                bar_colors.append("lightgray" if result_type == "shuffle" else color_cycle[j % len(color_cycle)])
            x_pos = np.arange(len(results))
            ax.bar(x_pos, means, bar_width, yerr=errors, capsize=5, color=bar_colors)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=15, ha="right")
            ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8)
            ax.set_ylim(0.3, 1.0)
            ax.set_ylabel("ROC AUC")
            ax.set_title(f"[{self.axis_map[train_key]}] Train: {e1} vs {e2}\n→ Test: {te1} vs {te2}")
        plt.suptitle("2x2 Cross-generalization decoder (time-averaged)")
        plt.tight_layout()
        plt.show()


def plot_cross_gen_weights(result, start=None, stop=None):
    """Plot average classifier weights per PC from a cross-generalization decoder result.

    Works with both cross_gen_results and cross_gen_2x2_results.
    Weights are averaged across folds and the specified time window.

    For LinearSVC  : plots signed coef_ weights.
    For BaggingClassifier (RF): plots unsigned feature_importances_.

    One subplot per training pair. PCs are on the y-axis, weight magnitude on the x-axis.

    Args:
        result : cross_gen_results or cross_gen_2x2_results
        start  : float, seconds from event onset to begin averaging (default: full epoch)
        stop   : float, seconds from event onset to stop averaging (default: full epoch)
    """
    if not hasattr(result, "models") or not result.models:
        raise ValueError("No models found on result object.")

    train_keys = [k for k, v in result.models.items() if v is not None]
    if not train_keys:
        raise ValueError("Models were not stored — re-run the decoder.")

    n_plots = len(train_keys)

    # determine figure height from the model with the most PCs
    def _n_pcs_for_key(key):
        m = result.models[key][0][0]
        return m.coef_.shape[1] if hasattr(m, "coef_") else len(m.feature_importances_)

    max_n_pcs = max(_n_pcs_for_key(k) for k in train_keys)

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, max(4, max_n_pcs * 0.45 + 2)),
                             squeeze=False)
    axes = axes[0]

    x_axis = np.linspace(-result.pre_window, result.event_length + result.post_window,
                         len(result.models[train_keys[0]]))

    t_start = int(np.searchsorted(x_axis, start)) if start is not None else 0
    t_stop  = int(np.searchsorted(x_axis, stop))  if stop  is not None else len(x_axis)

    for ax, train_key in zip(axes, train_keys):
        models_by_time = result.models[train_key]  # list[T] of list[fold]

        all_weights = []
        for t in range(t_start, t_stop):
            for model in models_by_time[t]:
                if hasattr(model, "coef_"):
                    all_weights.append(model.coef_[0])
                elif hasattr(model, "feature_importances_"):
                    all_weights.append(model.feature_importances_)

        if not all_weights:
            ax.set_title(f"Train: {train_key}\n(no models in window)")
            continue

        avg_weights = np.mean(all_weights, axis=0)   # (n_PCs,)
        sem_weights = sem(np.array(all_weights), axis=0)

        n_pcs = len(avg_weights)
        signed = hasattr(models_by_time[0][0], "coef_")
        colors = ["steelblue" if w >= 0 else "tomato" for w in avg_weights]
        y_pos = np.arange(n_pcs)
        pc_labels = [f"PC {i + 1}" for i in range(n_pcs)]

        ax.barh(y_pos, avg_weights, xerr=sem_weights, color=colors,
                capsize=3, error_kw={"linewidth": 0.8})
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pc_labels)
        ax.invert_yaxis()   # PC 1 at top
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_xlabel("Weight" if signed else "Feature importance")
        ax.set_title(f"Train: {train_key}")

    plt.suptitle("Cross-generalization decoder weights (avg ± SEM across folds/timebins)")
    plt.tight_layout()
    plt.show()


def plot_weights(result, start=None, stop=None):
    """Plot average classifier weights per PC from a trial_decoder result (all_results).

    One subplot per event. PCs on the y-axis, weight magnitude on the x-axis.
    Weights are averaged across folds and the specified time window.

    For LinearSVC  : plots signed coef_ weights.
    For BaggingClassifier (RF): plots unsigned feature_importances_.

    Args:
        result : all_results (returned by trial_decoder)
        start  : float, seconds from event onset to begin averaging (default: full epoch)
        stop   : float, seconds from event onset to stop averaging (default: full epoch)
    """
    events = result.events
    n_plots = len(events)

    # determine n_PCs and whether weights are signed from the first available model
    sample_models = result.results[events[0]].models
    sample_model = sample_models[0][0]
    if hasattr(sample_model, "coef_"):
        n_pcs = sample_model.coef_.shape[1]
        signed = True
    else:
        n_pcs = len(sample_model.feature_importances_)
        signed = False

    x_axis = np.linspace(
        -result.pre_window,
        result.event_length + result.post_window,
        len(sample_models),
    )
    t_start = int(np.searchsorted(x_axis, start)) if start is not None else 0
    t_stop  = int(np.searchsorted(x_axis, stop))  if stop  is not None else len(x_axis)

    pc_labels = [f"PC {i + 1}" for i in range(n_pcs)]
    height_fig = math.ceil(n_plots / 2)
    fig, axes = plt.subplots(height_fig, 2, figsize=(10, max(4, n_pcs * 0.45 + 2) * height_fig),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax, event in zip(axes_flat, events):
        models_by_time = result.results[event].models  # list[T] of list[fold]

        all_weights = []
        for t in range(t_start, t_stop):
            for model in models_by_time[t]:
                if hasattr(model, "coef_"):
                    all_weights.append(model.coef_[0])
                elif hasattr(model, "feature_importances_"):
                    all_weights.append(model.feature_importances_)

        if not all_weights:
            ax.set_title(f"{event}\n(no models in window)")
            continue

        avg_weights = np.mean(all_weights, axis=0)   # (n_PCs,)
        sem_weights = sem(np.array(all_weights), axis=0)

        colors = ["steelblue" if w >= 0 else "tomato" for w in avg_weights]
        y_pos = np.arange(n_pcs)

        ax.barh(y_pos, avg_weights, xerr=sem_weights, color=colors,
                capsize=3, error_kw={"linewidth": 0.8})
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pc_labels)
        ax.invert_yaxis()   # PC 1 at top
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_xlabel("Weight" if signed else "Feature importance")
        ax.set_title(event)

    # hide any unused subplots
    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    plt.suptitle("Decoder weights (avg ± SEM across folds/timebins)")
    plt.tight_layout()
    plt.show()
