import numpy as np

def evaluate_binary_classifier(model, x, y):
    yhat = model.feed_forward(x)
    yhatmax = (yhat == yhat.max(axis=1, keepdims = True)).astype(int)
    e = (2 * y )+ yhatmax
    tp = (e[:, 1] == 3).astype(int).sum()
    tn = (e[:, 1] == 0).astype(int).sum()
    fn = (e[:, 1] == 2).astype(int).sum()
    fp = (e[:, 1] == 1).astype(int).sum()
    return tp, fp, tn, fn


def evaluate_nonbinary_classifier(model, x, y):
    yhat = model.feed_forward(x)
    yhatmax = (yhat == yhat.max(axis=1, keepdims = True)).astype(int)
    metrics = []
    for col in range(y.shape[1]):
        yy = y[:, col]
        yyhatmax = yhatmax[:, col]
        e = 2 * yy + yyhatmax
        tp = (e == 3).astype(int).sum()
        tn = (e == 0).astype(int).sum()
        fn = (e == 2).astype(int).sum()
        fp = (e == 1).astype(int).sum()
        metrics.append((tp, fp, tn, fn))
    return metrics


def calculate_metrics(tp, fp, tn, fn):
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = 2.0 * (sensitivity * precision) / (sensitivity + precision)
    return (sensitivity, specificity, precision, f1)


def print_metrics(tp, fp, tn, fn):
    sensitivity, specificity, precision, f1 = calculate_metrics(tp, fp, tn, fn)
    print(f"{sensitivity = :.3f}, {specificity = :.3f}, {precision = :.3f}, {f1 = :.3f}\n")

def calculate_and_display_metrics(model, x, y):
    '''
        Specific to a binary classifier
    '''
    tp, fp, tn, fn = evaluate_binary_classifier(model, x, y)
    print_metrics(tp, fp, tn, fn)
