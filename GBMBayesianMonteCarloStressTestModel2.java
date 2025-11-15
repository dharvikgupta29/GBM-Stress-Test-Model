import java.util.Random;

/**
 * GBMBayesianMonteCarloStressTest (Hybrid Model)
 *
 * • Geometric Brownian Motion (GBM) with adaptive market behavior • Four
 * downside shocks (recession, war, pandemic, black-swan) • One upside "bull"
 * shock (positive recovery bursts) • Monthly pension out-flow (liabilities) •
 * Bayesian (Laplace-smoothed) posterior of shock frequencies, updated
 * adaptively during Monte-Carlo runs, with per-run reset + clamping
 *
 * Hybrid features: - Realistic drift/volatility for a diversified pension-style
 * portfolio - Realistic but rare shocks - Cap on extreme monthly loss (25%) -
 * Gentler Bayesian adaptation (alpha = 0.002) - Shock probabilities reset each
 * simulation + clamped into reasonable bands
 *
 * @author Dharvik Gupta
 */
public final class GBMBayesianMonteCarloStressTest {

    /** Private constructor: class may not be instantiated. */
    private GBMBayesianMonteCarloStressTest() {
    }

    /** Clamp helper. */
    private static double clamp(double x, double lo, double hi) {
        if (x < lo) {
            return lo;
        } else if (x > hi) {
            return hi;
        }
        return x;
    }

    /** Entry-point. */
    public static void main(String[] args) {

        // 1. FUND & MODEL SETTINGS
        final double initialFund = 1_000_000_000; // $1 B
        final double baseDrift = 0.07; // 7% per annum
        final double baseVolatility = 0.10; // 10% per annum
        final int years = 10;
        final int months = years * 12;
        final int runs = 1_000_000;
        final double floor = 700_000_000; // Insolvency threshold
        final double monthlyPayout = 3_000_000; // Outflow per month
        final double dt = 1.0 / 12.0; // Monthly step

        // 2. BASE SHOCK PROBABILITIES (monthly) - reset per simulation
        final double baseProbRec = 0.0020; // recession
        final double baseProbWar = 0.0005; // war
        final double baseProbPan = 0.0005; // pandemic
        final double baseProbBsw = 0.0003; // black swan
        final double baseProbBull = 0.0400; // bull “burst”

        // 3. SHOCK MAGNITUDES (sampled U[0, max])
        final double maxRecDrop = 0.04; // up to -4%
        final double maxWarDrop = 0.03; // up to -3%
        final double maxPanDrop = 0.06; // up to -6%
        final double maxBswDrop = 0.10; // up to -10%
        final double maxBullGain = 0.03; // up to +3%
        final double maxMonthlyLoss = 0.25; // max -25% in any month (all shocks combined)

        // 4. Random engine
        Random rng = new Random();

        // 5. Results
        int insolventRuns = 0;
        double totalFinalValue = 0.0;

        int totalRecHits = 0, totalWarHits = 0, totalPanHits = 0, totalBswHits = 0;

        // Bayesian learning rate (gentle)
        final double alpha = 0.002;

        // 6. Monte Carlo Simulations
        for (int run = 0; run < runs; run++) {

            // Reset probabilities at the start of each simulation
            double probRec = baseProbRec;
            double probWar = baseProbWar;
            double probPan = baseProbPan;
            double probBsw = baseProbBsw;
            double probBull = baseProbBull;

            double fund = initialFund;
            int recHits = 0, warHits = 0, panHits = 0, bswHits = 0;

            double drift = baseDrift;
            double volatility = baseVolatility;

            for (int month = 0; month < months; month++) {

                // Adaptive volatility scaling with fund health (tighter band)
                double healthFactor = fund / initialFund;
                healthFactor = Math.max(0.7, Math.min(1.3, healthFactor));

                volatility = baseVolatility * (1.0 + (1.0 - healthFactor) * 0.4);

                // Drift mean-reverts, softer adjustment
                drift = baseDrift * (0.90 + 0.20 * healthFactor);

                // GBM base return
                double driftMain = (drift - 0.5 * volatility * volatility) * dt;
                double volShock = volatility * Math.sqrt(dt) * rng.nextGaussian();
                double rtn = Math.exp(driftMain + volShock) - 1.0;

                // Sum downside shocks first, then cap the total monthly loss
                double totalLoss = 0.0;

                if (rng.nextDouble() < probRec) {
                    totalLoss += rng.nextDouble() * maxRecDrop;
                    recHits++;
                }
                if (rng.nextDouble() < probWar) {
                    totalLoss += rng.nextDouble() * maxWarDrop;
                    warHits++;
                }
                if (rng.nextDouble() < probPan) {
                    totalLoss += rng.nextDouble() * maxPanDrop;
                    panHits++;
                }
                if (rng.nextDouble() < probBsw) {
                    totalLoss += rng.nextDouble() * maxBswDrop;
                    bswHits++;
                }

                // Cap combined loss
                totalLoss = Math.min(totalLoss, maxMonthlyLoss);
                rtn -= totalLoss;

                // Bull market gain
                if (rng.nextDouble() < probBull) {
                    rtn += rng.nextDouble() * maxBullGain;
                }

                // Apply return and liability outflow
                fund *= (1.0 + rtn);
                fund -= monthlyPayout;

                if (fund <= 0.0) {
                    fund = 0.0;
                    break;
                }

                // Adaptive Bayesian shock reweighting (Laplace-style, per simulation)
                probRec = probRec * (1.0 - alpha)
                        + alpha * ((recHits + 1.0) / (month + 2.0));
                probWar = probWar * (1.0 - alpha)
                        + alpha * ((warHits + 1.0) / (month + 2.0));
                probPan = probPan * (1.0 - alpha)
                        + alpha * ((panHits + 1.0) / (month + 2.0));
                probBsw = probBsw * (1.0 - alpha)
                        + alpha * ((bswHits + 1.0) / (month + 2.0));

                // Clamp probabilities to keep them in realistic ranges
                probRec = clamp(probRec, 0.0001, 0.0200); // 0.01%–2.0%
                probWar = clamp(probWar, 0.00005, 0.0100); // 0.005%–1.0%
                probPan = clamp(probPan, 0.00005, 0.0100);
                probBsw = clamp(probBsw, 0.00001, 0.0050); // super rare
                // Bull probability left constant for simplicity (could also adapt if desired)
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

        System.out.println("\nTotal Shock Events Across All Runs:");
        System.out.printf("  Recession : %,d%n", totalRecHits);
        System.out.printf("  War       : %,d%n", totalWarHits);
        System.out.printf("  Pandemic  : %,d%n", totalPanHits);
        System.out.printf("  Black Swan: %,d%n", totalBswHits);
    }
}
