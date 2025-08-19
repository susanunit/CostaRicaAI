# Tuesday notes

## Building bins and data insights
```
df.groupby('IssueReported').size().sort_values(ascending=False).head(10)

df['Date'] = pd.to_datetime(df['Published Date'])
df['Month'] = df['Date'].dt.month
monthly_counts = df.groupby('Month').size()
```
---

## Cleaning up Longitude and Latitude
```
df = df[(df['Latitude'] != 0)]
df = df[(df['Longitude'] != 0)]
df = df[(df['Latitude'] <= 35)]
```

