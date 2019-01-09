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
