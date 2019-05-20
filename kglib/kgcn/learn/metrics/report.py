#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.
#

import sklearn.metrics as metrics
import numpy as np


def format_list(list_to_format, formatting="%.2f"):
    return [formatting % e for e in list_to_format]


def multilabel_confusion_matrix(labels, predictions):
    """Generate multilabel confusion matrix by computing multiple binary
    confusion matrices for each class
    https://stackoverflow.com/questions/53886370/multi-class-multi-label-confusion-matrix-with-sklearn

    Args:
        labels (numpy.ndarray): ground truth
        predictions (numpy.ndarray): model predictions
    """
    if isinstance(labels, list):
        labels = np.array(labels)
    if isinstance(predictions, list):
        predictions = np.array(predictions)

    assert isinstance(labels, np.ndarray) and \
        isinstance(predictions, np.ndarray), \
        "labels and predictions must be numpy arrays"

    num_classes = labels.shape[1]
    conf_mat_dict = {}

    for label_col in range(num_classes):
        actual_labels = labels[:, label_col]
        predicted_labels = predictions[:, label_col]
        conf_mat_dict[label_col] = metrics.confusion_matrix(
            y_pred=predicted_labels, y_true=actual_labels)

    for label, matrix in conf_mat_dict.items():
        print("Confusion matrix for label {}:".format(label))
        print(matrix)


def report_multiclass_metrics(labels, predictions):
    skl_cm = metrics.confusion_matrix(labels, predictions)
    print('Confusion matrix')
    print(skl_cm)

    class_precisions = metrics.precision_score(
        labels, predictions, average=None)
    print(f'Class precisions:   {format_list(class_precisions)}')

    class_recalls = metrics.recall_score(labels, predictions, average=None)
    print(f'Class recalls:      {format_list(class_recalls)}')

    class_f1s = metrics.f1_score(labels, predictions, average=None)
    print(f'Class F1-scores:    {format_list(class_f1s)}')

    class_accuracies = metrics.accuracy_score(labels, predictions)
    print(f'Accuracy:           {class_accuracies:.2f}')


def report_multilabel_metrics(labels, predictions):
    if isinstance(labels, list):
        labels = np.array(labels)
    if isinstance(predictions, list):
        predictions = np.array(predictions)

    assert isinstance(labels, np.ndarray) and \
        isinstance(predictions, np.ndarray), \
        "labels and predictions must be numpy arrays"

    multilabel_confusion_matrix(labels, predictions)

    class_precisions = metrics.precision_score(
        labels, predictions, average=None)
    print(f'Class precisions:      {format_list(class_precisions)}')

    class_recalls = metrics.recall_score(labels, predictions, average=None)
    print(f'Class recalls:         {format_list(class_recalls)}')

    class_f1s = metrics.f1_score(labels, predictions, average=None)
    print(f'Class F1-scores:       {format_list(class_f1s)}')

    class_accuracies = metrics.hamming_loss(labels, predictions)
    print(f'Avg. class Accuracy:   {np.mean(class_accuracies):.2f}')

    exact_match_score = metrics.accuracy_score(labels, predictions)
    print(f'Exact match score:     {exact_match_score:.2f}')
