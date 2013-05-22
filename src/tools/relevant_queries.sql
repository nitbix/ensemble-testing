-- Show all results

select technique, aggregation, problem, ensemble_size,
count(*) as runs,
min(misclassified_samples) as ms_min,
avg(misclassified_samples) as ms_avg,
max(misclassified_samples) as ms_max,
avg(micro_precision) as avg_prec,
avg(micro_recall) as avg_recall
from chains c
join runs r on r.chain = c.id
where invalidated = 0 and is_test=1
group by technique, problem, ensemble_size, aggregation
order by problem, technique, aggregation, ensemble_size