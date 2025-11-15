import java.util.Random;

/**
 * GBMBayesianMonteCarloStressTest
 *
 * • Geometric Brownian Motion (GBM) → adaptive market behavior • Four downside
 * shocks (recession, war, pandemic, black-swan) • One upside “bull” shock
 * (positive recovery bursts) • Monthly pension out-flow (liabilities) •
 * Bayesian (Laplace-smoothed) posterior of shock frequencies, updated
 * adaptively during Monte-Carlo runs
 *
 * @author Dharvik Gupta
 */
public final class GBMBayesianMonteCarloStressTest {

    /** Private constructor: class may not be instantiated. */
    private GBMBayesianMonteCarloStressTest() {
    }

    /** Entry-point. */
    public static void main(String[] args) {

        // 1. FUND & MODEL SETTINGS
        final double initialFund = 1_000_000_000; // $1 B
        final double baseDrift = 0.06; // 6% per annum
        final double baseVolatility = 0.12; // 12% per annum
        final int years = 10;
        final int months = years * 12;
        final int runs = 1_000_000;
        final double floor = 700_000_000; // Insolvency threshold
        final double monthlyPayout = 3_000_000; // Outflow per month
        final double dt = 1.0 / 12.0; // Monthly step

        // 2. BASE SHOCK PROBABILITIES (monthly)
        double probRec = 0.003;
        double probWar = 0.001;
        double probPan = 0.002;
        double probBsw = 0.002;
        double probBull = 0.03;

        // 3. SHOCK MAGNITUDES (sampled U[0, max])
        final double maxRecDrop = 0.06;
        final double maxWarDrop = 0.04;
        final double maxPanDrop = 0.08;
        final double maxBswDrop = 0.12;
        final double maxBullGain = 0.04;

        // 4. Random engine
        Random rng = new Random();

        // 5. Results
        int insolventRuns = 0;
        double totalFinalValue = 0;

        int totalRecHits = 0, totalWarHits = 0, totalPanHits = 0, totalBswHits = 0;

        // 6. Monte Carlo Simulations
        for (int run = 0; run < runs; run++) {
            double fund = initialFund;
            int recHits = 0, warHits = 0, panHits = 0, bswHits = 0;

            double drift = baseDrift;
            double volatility = baseVolatility;

            for (int month = 0; month < months; month++) {

                // Adaptive volatility scaling with fund health
                double healthFactor = Math.max(0.5, Math.min(1.5, fund / initialFund));
                volatility = baseVolatility * (1.0 + (1.0 - healthFactor) * 0.5);

                // Drift mean-reverts: higher when fund < baseline
                drift = baseDrift * (0.8 + 0.4 * healthFactor);

                // GBM base return
                double driftMain = (drift - 0.5 * volatility * volatility) * dt;
                double volShock = volatility * Math.sqrt(dt) * rng.nextGaussian();
                double rtn = Math.exp(driftMain + volShock) - 1.0;

                // Downside shocks
                if (rng.nextDouble() < probRec) {
                    rtn -= rng.nextDouble() * maxRecDrop;
                    recHits++;
                }
                if (rng.nextDouble() < probWar) {
                    rtn -= rng.nextDouble() * maxWarDrop;
                    warHits++;
                }
                if (rng.nextDouble() < probPan) {
                    rtn -= rng.nextDouble() * maxPanDrop;
                    panHits++;
                }
                if (rng.nextDouble() < probBsw) {
                    rtn -= rng.nextDouble() * maxBswDrop;
                    bswHits++;
                }

                // Bull market gain
                if (rng.nextDouble() < probBull) {
                    rtn += rng.nextDouble() * maxBullGain;
                }

                // Apply return and liability outflow
                fund *= (1 + rtn);
                fund -= monthlyPayout;

                // Adaptive Bayesian shock reweighting (Laplace-style)
                double alpha = 0.01; // learning rate for adaptation
                double totalShocks = recHits + warHits + panHits + bswHits + 1e-9;
                probRec = (probRec * (1 - alpha))
                        + alpha * ((recHits + 1.0) / (month + 2.0));
                probWar = (probWar * (1 - alpha))
                        + alpha * ((warHits + 1.0) / (month + 2.0));
                probPan = (probPan * (1 - alpha))
                        + alpha * ((panHits + 1.0) / (month + 2.0));
                probBsw = (probBsw * (1 - alpha))
                        + alpha * ((bswHits + 1.0) / (month + 2.0));

                if (fund <= 0) {
                    fund = 0;
                    break;
                }
            }

            if (fund < floor) {
                insolventRuns++;
            }

            totalFinalValue += fund;
            totalRecHits += recHits;
            totalWarHits += warHits;
            totalPanHits += panHits;
            totalBswHits += bswHits;
        }

        // 7. Bayesian posterior (Laplace smoothing)
        double denom = runs * months + 2.0;
        double postRec = (totalRecHits + 1.0) / denom;
        double postWar = (totalWarHits + 1.0) / denom;
        double postPan = (totalPanHits + 1.0) / denom;
        double postBsw = (totalBswHits + 1.0) / denom;

        // 8. Output
        double avgFinal = totalFinalValue / runs;
        double insolvencyRisk = 100.0 * insolventRuns / runs;

        System.out.printf("Simulations Run        : %,d%n", runs);
        System.out.printf("Average Final Value    : $%,.0f%n", avgFinal);
        System.out.printf("Insolvency Probability : %.2f%%%n", insolvencyRisk);

        System.out.println("\nPosterior Shock Probabilities (monthly):");
        System.out.printf("  Recession : %.4f%n", postRec);
        System.out.printf("  War       : %.4f%n", postWar);
        System.out.printf("  Pandemic  : %.4f%n", postPan);
        System.out.printf("  Black Swan: %.4f%n", postBsw);
    }
}
