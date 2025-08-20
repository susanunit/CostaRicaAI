# Notas del Miercoles

## Python Notebook courtesy of Luis Carlos Navarro Todd.

[Python Notebook](./Supervised_Unsupervised.ipynb)

Notice the 'percentage' function and call:

```
def percentage(df, x_label, y_label):
    labels = df[y_label].unique()
    for x in df[x_label].unique().tolist():
        subdataset = df[df[x_label] == x]
        totals = subdataset[y_label].value_counts().tolist()
        results = list(map(lambda x: x/sum(totals) ,totals))
        if len(totals) != 2:
            continue
        print(f"{x:27} | {labels[0]:7}: {results[0]:.3f} | {labels[1]:5}: {results[1]:.3f}")
        
percentage(df, "Issue Reported", "road_type")
```

