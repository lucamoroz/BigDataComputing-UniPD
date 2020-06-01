import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;

import java.util.*;

public class DiversityMaximization {

    private final static Random RAND = new Random(123);

    public static void main(String... args) {

        long startTime, endTime, duration;

        String dataFolderPath = "data/uber-medium.csv";
        int k = 5;
        int L = 4;

        SparkConf conf = new SparkConf(true).setAppName("Homework1 - ClassCount");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        startTime = System.currentTimeMillis();
        JavaRDD<Vector> inputPoints = sc.textFile(dataFolderPath).map(DiversityMaximization::strToVector).repartition(L).cache();
        long numPoints = inputPoints.count();
        endTime = System.currentTimeMillis();
        duration = endTime - startTime;

        System.out.println("Number of points = " + numPoints);
        System.out.println("k = " + k);
        System.out.println("L = " + L);
        System.out.println("Initialization time = " + duration);

        ArrayList<Vector> points = runMapReduce(inputPoints, k, L);

        double avgDistance = measure(points);
    }

    public static ArrayList<Vector> runMapReduce(JavaRDD<Vector> pointsRDD, int k, int L) {
        long startTime, endTime, duration;

        // ROUND 1 - FFT on each partition
        startTime = System.currentTimeMillis();

        // Force computation
        ArrayList<Vector> pts = new ArrayList<>(
                pointsRDD.mapPartitions((points) -> {
                    ArrayList<Vector> S = new ArrayList<>();
                    while (points.hasNext()) {
                        S.add(points.next());
                    }

                    return FarthestFirstTraversal(S, k).iterator();
                }).collect()
        );

        endTime = System.currentTimeMillis();
        duration = endTime - startTime;
        System.out.println("Runtime of Round 1 = " + duration);

        // ROUND 2 - 2-approx algorithm
        startTime = System.currentTimeMillis();

        pts = runSequential(pts, k);

        endTime = System.currentTimeMillis();
        duration = endTime - startTime;
        System.out.println("Runtime of Round 2 = " + duration);

        return pts;
    }

    public static double measure(ArrayList<Vector> pointSet) {
        double tot = 0;
        for (int i = 0; i < pointSet.size(); i++) {
            for (int j = i; j < pointSet.size(); j++)
                tot += Math.sqrt(Vectors.sqdist(pointSet.get(i), pointSet.get(j)));
        }

        return tot / (pointSet.size() * (pointSet.size() - 1) * 0.5);
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // METHOD runSequential
    // Sequential 2-approximation based on matching
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    public static ArrayList<Vector> runSequential(final ArrayList<Vector> points, int k) {

        final int n = points.size();
        if (k >= n) {
            return points;
        }

        ArrayList<Vector> result = new ArrayList<>(k);
        boolean[] candidates = new boolean[n];
        Arrays.fill(candidates, true);
        for (int iter=0; iter<k/2; iter++) {
            // Find the maximum distance pair among the candidates
            double maxDist = 0;
            int maxI = 0;
            int maxJ = 0;
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    for (int j = i+1; j < n; j++) {
                        if (candidates[j]) {
                            // Use squared euclidean distance to avoid an sqrt computation!
                            double d = Vectors.sqdist(points.get(i), points.get(j));
                            if (d > maxDist) {
                                maxDist = d;
                                maxI = i;
                                maxJ = j;
                            }
                        }
                    }
                }
            }
            // Add the points maximizing the distance to the solution
            result.add(points.get(maxI));
            result.add(points.get(maxJ));
            // Remove them from the set of candidates
            candidates[maxI] = false;
            candidates[maxJ] = false;
        }
        // Add an arbitrary point to the solution, if k is odd.
        if (k % 2 != 0) {
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    result.add(points.get(i));
                    break;
                }
            }
        }
        if (result.size() != k) {
            throw new IllegalStateException("Result of the wrong size");
        }
        return result;

    } // END runSequential

    public static ArrayList<Vector> FarthestFirstTraversal(ArrayList<Vector> S, int k) {
        if (k > S.size())
            throw new IllegalArgumentException("Assertion k < |S| failed");

        // holds selected centers
        ArrayList<Vector> C = new ArrayList<>();

        // randomly select the first center
        int lastCenter = RAND.nextInt(S.size());
        C.add(S.get(lastCenter));

        double[] pointCDistance = new double[S.size()];
        for (int i = 0; i < S.size(); i++)
            pointCDistance[i] = Vectors.sqdist(S.get(lastCenter), S.get(i));

        for (int i = 1; i < k; i++) {

            double currMaxDistance = -1;
            int nextCenter = -1;

            for (int j = 0; j < S.size(); j++) {
                // Check if the last selected center is closer to p. If so, update closest center distance for p
                // Note that if j == lastCenter then its updatedDist is set to zero
                double updatedDist = Vectors.sqdist(S.get(j), S.get(lastCenter));
                if (updatedDist < pointCDistance[j])
                    pointCDistance[j] = updatedDist;

                // Check if p is the farthest point from C (ie if p could be the next center)
                if (pointCDistance[j] > currMaxDistance) {
                    currMaxDistance = pointCDistance[j];
                    nextCenter = j;
                }
            }

            C.add(S.get(nextCenter));
            lastCenter = nextCenter;
        }
        return C;
    }

    public static org.apache.spark.mllib.linalg.Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }
}
