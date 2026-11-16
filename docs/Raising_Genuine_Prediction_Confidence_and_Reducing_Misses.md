# Raising Genuine Prediction Confidence and Reducing Misses

Prepared 13 Nov 2026 for the Market Sentiment Agent project.

## Summary

Confidence cannot be pushed up by adjusting the displayed score. A 70% confidence should mean the call is right about 70% of the time. If the figure is inflated, traders are misled and an advisory product takes on compliance risk. What can be improved is the share of situations where high confidence is genuinely justified, and confidence should be shown only there.

## 1. Why confidence sits near 50% today

Part of it is honest: intraday direction is a weak signal. Part of it is by design. The current documentation shrinks probabilities toward the base rate to stay within 35 to 65%. That shrinkage is the right choice while data is thin, but it means displayed confidence cannot exceed 65% right now. The way out is more information, then recalibration, and not removing the shrinkage.

## 2. What increases genuine confidence

### Confluence of independent evidence

Confidence should rise when unrelated signals agree:

- **Multi-timeframe alignment:** 5m, 15m, 60m and daily trend pointing the same way.
- **Participation confirmation:** breadth, volume and open interest, and the heavyweight stocks all supporting the index move.
- **Options positioning:** put-call ratio, open-interest shifts and implied volatility agreeing with the price move.
- **Key-level context:** room before the previous day's high or low, VWAP and opening-range levels.
- **Cross-asset agreement:** USD/INR, crude oil and global futures not contradicting the call.

### Know when predictability is higher

Let the model learn confidence by condition:

- Time of day (the first 15 to 30 minutes are noisy).
- Volatility expansion after compression.
- Expiry days (the backtest showed 61.9% on 84 windows).
- Scheduled event windows (RBI, results, US data), where confidence may need to be lowered or withheld.

### Measure uncertainty explicitly

- **Ensemble disagreement:** if the strategies and models disagree, confidence drops automatically.
- **Meta-labeling:** train a second model to predict whether the first model's call will be correct, and use its output as the confidence.
- **Conformal prediction:** gives prediction sets with statistical coverage guarantees. When the set contains both "up" and "down", the system abstains.

### Calibrate per regime

Use isotonic or Platt scaling once there is enough data, so a 65% in a trending regime and in a sideways regime both mean 65%.

### More data

With 1 to 2 years of history, confidence estimates stop being dominated by noise.

## 3. Selective prediction: where "fewer misses" comes from

Fewer misses come from abstaining, not from a better score on every window. Plot a coverage-accuracy curve: as the system restricts to higher-confidence calls, accuracy should rise while coverage falls. For example, the system might act on 15 to 20% of windows at 60 to 65% accuracy and say "no clear view" otherwise.

The current high-conviction tier (61.4% on 140 calls) is an early sign of this, but each tier needs several hundred calls before the number can be trusted.

High confidence will be rare, and misses should not be expected to nearly vanish.

## 4. Reduce the cost of a miss

- Always show the invalidation level, so a miss costs a small, defined amount.
- Tie confidence tiers to risk guidance (for example, smaller exposure at lower tiers).
- Report average loss on misses against average gain on hits, not just hit rate.

## 5. Proposed confidence framework

Combine four inputs into a tier of high, medium, low or no view:

1. Calibrated probability
2. Signal agreement across strategies and timeframes
3. Regime clarity
4. Event and data-quality checks

Validate that each tier's observed hit rate matches what it claims, using the same purged walk-forward protocol as the rest of the project.

## 6. Next steps

- Add this as Phase 3b in the implementation plan.
- Build the confidence-tier code and the coverage-accuracy report.
- Collect enough live shadow predictions per tier to verify the claimed hit rates before any tier is shown to traders.
