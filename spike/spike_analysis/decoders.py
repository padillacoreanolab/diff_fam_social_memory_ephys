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


def _flat_decoder(spike_collection, num_fold, no_PCs, events, event_length, pre_window, post_window,
                  min_neurons, condition_dict, decoder_data, classifier_type, **kwargs):
    """Run one classifier per event on full flattened trial (T * n_PCs features) instead of per timebin."""
    if decoder_data is None:
        decoder_data = trial_PCA(
            spike_collection, event_length, pre_window, post_window, no_PCs,
            events=events, min_neurons=min_neurons, condition_dict=condition_dict,
        )
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
    return all_results(results_dict, shuffle_results_dict, num_fold, event_labels, event_length, pre_window, post_window)


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
    input="timebin",
    classifier_type="RF",
    **kwargs,
):
    if LOO:
        return _trial_decoder_LOO(
            spike_collection, no_PCs, events, event_length, pre_window, post_window, min_neurons, condition_dict,
            classifier_type=classifier_type, input=input, **kwargs,
        )
    if input == "full_trial":
        return _flat_decoder(
            spike_collection, num_fold, no_PCs, events, event_length, pre_window, post_window,
            min_neurons, condition_dict, decoder_data, classifier_type, **kwargs,
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
            if classifier_type == "linear":
                results, shuffle_results = _linear(t_data, labels, num_fold, **kwargs)
            else:
                results, shuffle_results = _random_forest(t_data, labels, num_fold)
            results_dict[event].append(results)
            shuffle_results_dict[event].append(shuffle_results)
        if len(events) == 2:
            break
    result_object = all_results(
        results_dict, shuffle_results_dict, num_fold, event_labels, event_length, pre_window, post_window
    )
    return result_object


def _trial_decoder_LOO(spike_collection, no_PCs, events, event_length, pre_window, post_window, min_neurons, condition_dict, classifier_type="RF", input="timebin", decoder_data=None, recording_labels=None, **kwargs):
    if decoder_data is None or recording_labels is None:
        decoder_data, recording_labels = trial_PCA(
            spike_collection, event_length, pre_window, post_window, no_PCs,
            events=events, min_neurons=min_neurons, condition_dict=condition_dict,
            return_recording_labels=True,
        )
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
    no_PCs,
    events,
    event_length,
    pre_window=0,
    post_window=0,
    min_neurons=0,
    condition_dict=None,
    plot=True,
    LOO=False,
    input="timebin",
    classifier_type="RF",
    perm_type="random",
    **kwargs,
):
    decoder_data, recording_labels = trial_PCA(
        spike_collection, event_length, pre_window, post_window, no_PCs,
        events=events, min_neurons=min_neurons, condition_dict=condition_dict,
        return_recording_labels=True,
    )
    shuffled_data, shuffled_labels = _within_subject_shuffle(
        decoder_data, recording_labels, events, perm_type
    )
    if LOO:
        return _trial_decoder_LOO(
            spike_collection, no_PCs, events, event_length, pre_window, post_window,
            min_neurons, condition_dict,
            classifier_type=classifier_type, input=input,
            decoder_data=shuffled_data, recording_labels=shuffled_labels,
            **kwargs,
        )
    return trial_decoder(
        spike_collection, num_fold, no_PCs, events, event_length,
        pre_window=pre_window, post_window=post_window, min_neurons=min_neurons,
        condition_dict=condition_dict, decoder_data=shuffled_data,
        plot=plot, LOO=False, input=input, classifier_type=classifier_type,
        **kwargs,
    )


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


def _cross_gen_decoder(decoder_data, events, num_fold, classifier_type, **kwargs):
    """Core cross-generalization decoder.

    For each pairwise training set (e1, e2), trains a classifier per fold per timebin,
    then scores on two generalization test sets using the held-out event (e3).

    Label rule:
      - e1=0, e2=1 in training
      - Test (e1 vs e3): e1=0, e3=1  (e3 gets e2's label)
      - Test (e2 vs e3): e2=1, e3=0  (e3 gets e1's label)

    Returns:
        raw_results : {train_key: {test_key: list of T dicts with 'test_roc_auc': array(num_fold,)}}
    """
    assert len(events) == 3, "cross-generalization requires exactly 3 events"

    T = decoder_data[events[0]][0].shape[0]

    # split each event's trials into num_fold bins independently
    event_folds = {event: __split_into_folds__(decoder_data[event], num_fold) for event in events}

    raw_results = {}

    for e1, e2 in combinations(events, 2):
        e3 = next(e for e in events if e != e1 and e != e2)
        train_key = f"{e1}_{e2}"

        # (test_event1, test_event2, label_for_event1, label_for_event2)
        test_pairs = [
            (e1, e3, 0, 1),  # e3 gets e2's label (1)
            (e2, e3, 1, 0),  # e3 gets e1's label (0)
        ]

        # accumulators: test_key → (T, num_fold) array
        auc_arrays = {f"{te1}_{te2}": np.full((T, num_fold), np.nan) for te1, te2, _, _ in test_pairs}

        for k in range(num_fold):
            X_train_all, y_train = __build_train_fold__(k, event_folds, e1, e2)
            X_test_per_pair, y_test_per_pair = __build_test_fold__(k, event_folds, test_pairs)

            for t in range(T):
                clf = __fit_clf_single__(X_train_all[:, :, t], y_train, classifier_type, **kwargs)
                for te1, te2, _, _ in test_pairs:
                    test_key = f"{te1}_{te2}"
                    auc = __score_test__(
                        clf, X_test_per_pair[test_key][:, :, t], y_test_per_pair[test_key], classifier_type
                    )
                    auc_arrays[test_key][t, k] = auc

        # convert to list-of-T-dicts format (parallel to results_dict in trial_decoder)
        raw_results[train_key] = {}
        for test_key, auc_arr in auc_arrays.items():
            raw_results[train_key][test_key] = [
                {"test_roc_auc": auc_arr[t, :]} for t in range(T)
            ]

    return raw_results


def trial_decoder_cross_generalization(
    spike_collection,
    num_fold,
    no_PCs,
    events,
    event_length,
    pre_window=0,
    post_window=0,
    min_neurons=0,
    condition_dict=None,
    classifier_type="RF",
    **kwargs,
):
    """Cross-generalization decoder for exactly 3 events.

    Trains on each pairwise combination and tests generalization to the held-out event.
    See cross_gen_results for output structure.
    """
    assert len(events) == 3, "cross-generalization requires exactly 3 events"
    decoder_data = trial_PCA(
        spike_collection, event_length, pre_window, post_window, no_PCs,
        events=events, min_neurons=min_neurons, condition_dict=condition_dict,
    )
    raw_results = _cross_gen_decoder(decoder_data, events, num_fold, classifier_type, **kwargs)
    return cross_gen_results(raw_results, num_fold, event_length, pre_window, post_window)


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

    def __init__(self, raw_results, num_fold, event_length, pre_window, post_window):
        self.num_fold = num_fold
        self.event_length = event_length
        self.pre_window = pre_window
        self.post_window = post_window
        self.roc_auc_scores = {}
        for train_key, test_pairs in raw_results.items():
            self.roc_auc_scores[train_key] = {
                test_key: nested_model_result(timebin_list, num_fold)
                for test_key, timebin_list in test_pairs.items()
            }

    def __repr__(self):
        lines = [f"Cross-generalization decoder | {self.num_fold} folds"]
        for train_key, test_pairs in self.roc_auc_scores.items():
            lines.append(f"  Trained on {train_key}:")
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

        One subplot per training pair (3 total), two bars each — one per generalization
        test case. Mean and SEM are computed by first averaging over timebins in
        [start, stop], then taking mean ± SEM across folds.

        Args:
            start : float, seconds from event onset to start averaging (default 0)
            stop  : float, seconds from event onset to stop averaging (default event_length)
        """
        train_keys = list(self.roc_auc_scores.keys())
        plt.figure(figsize=(4 * len(train_keys), 4))
        bar_width = 0.3

        for i, train_key in enumerate(train_keys, 1):
            plt.subplot(1, len(train_keys), i)
            test_pairs = self.roc_auc_scores[train_key]
            test_keys = list(test_pairs.keys())

            means, errors = [], []
            for test_key in test_keys:
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

            x_pos = np.arange(len(test_keys))
            plt.bar(x_pos, means, bar_width, yerr=errors, capsize=5)
            plt.xticks(x_pos, test_keys, rotation=15, ha="right")
            plt.axhline(0.5, color="k", linestyle="--", linewidth=0.8)
            plt.ylim(0.3, 1.0)
            plt.ylabel("ROC AUC")
            plt.title(f"Train: {train_key}")

        plt.suptitle("Cross-generalization decoder (time-averaged)")
        plt.tight_layout()
        plt.show()
