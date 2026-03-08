import sys
sys.path.insert(0, '.')
from climate_resilience import ClimateDataLoader, ClimateResilienceIndex

loader = ClimateDataLoader(offline_mode=True)
raw_df = loader.generate_synthetic_dataset(n_zones=10, n_periods=4, seed=42)
norm_df = loader.normalize_indicators(raw_df)
latest = loader.get_latest_period(norm_df)

cri = ClimateResilienceIndex(context='sahel')
results = cri.compute_batch(latest)

meta = latest[['zone_id','zone_name','country','latitude','longitude','period']].drop_duplicates('zone_id')
full = results.merge(meta, on='zone_id', how='left')

print('results columns:', list(results.columns))
print('meta columns:   ', list(meta.columns))
print('full columns:   ', list(full.columns))
print('zone_id sample: ', results['zone_id'].iloc[0])