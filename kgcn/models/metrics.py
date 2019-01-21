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


def format_list(list_to_format, formatting="%.2f"):
    return [formatting % e for e in list_to_format]


def report_multiclass_metrics(labels, predictions):
    skl_cm = metrics.confusion_matrix(labels, predictions)
    print('Confusion matrix')
    print(skl_cm)

    class_precisions = metrics.precision_score(labels, predictions, average=None)
    print(f'Class precisions:   {format_list(class_precisions)}')

    class_recalls = metrics.recall_score(labels, predictions, average=None)
    print(f'Class recalls:      {format_list(class_recalls)}')

    class_f1s = metrics.f1_score(labels, predictions, average=None)
    print(f'Class F1-scores:    {format_list(class_f1s)}')

    class_accuracies = metrics.accuracy_score(labels, predictions)
    print(f'Accuracy:           {class_accuracies:.2f}')
