import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.IOException;
import java.util.*;

public class ClassCount {

    public static void main(String[] args) throws IOException {

        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        SparkConf conf = new SparkConf(true).setAppName("Homework1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // Read number of partitions
        int K = Integer.parseInt(args[0]);

        // Read input file and subdivide it into K random partitions
        JavaRDD<String> objects = sc.textFile(args[1]).repartition(K);

        JavaPairRDD<String, Long> count;

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // ClassCount with deterministic partitions
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        count = objects
                .mapToPair((object) -> {    // <-- MAP PHASE (R1)
                    // "i label" --> (i mod K, label)
                    String[] tokens = object.split(" ");
                    long index = Long.parseLong(tokens[0]);
                    String label = tokens[1];
                    return new Tuple2<>(index % K, label);
                })
                .groupByKey()    // <-- REDUCE PHASE (R1)
                .flatMapToPair((labels) -> {
                    // (key, [ label1, label2, ... ]) --> [ (label1, count1), (label2, count2), ... ]
                    HashMap<String, Long> counts = new HashMap<>();
                    for (String label : labels._2()) {
                        counts.put(label, counts.getOrDefault(label, 0L) + 1L);
                    }
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .reduceByKey(Long::sum); // <-- REDUCE PHASE (R2)

        System.out.println("VERSION WITH DETERMINISTIC PARTITIONS");
        System.out.print("Output pairs =");
        count
                .sortByKey()
                .collect()
                .forEach((x) -> System.out.print(" " + x.toString()));
        System.out.println();

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // ClassCount with spark partitions
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        count = objects
                .map((object) -> {    // <-- MAP PHASE (R1)
                    String[] tokens = object.split(" ");
                    return tokens[1];
                })
                .mapPartitionsToPair((partition) -> {    // <-- REDUCE PHASE (R1)
                    long partitionSize = 0;
                    HashMap<String, Long> counts = new HashMap<>();
                    while (partition.hasNext()){
                        String label = partition.next();
                        counts.put(label, counts.getOrDefault(label, 0L) + 1L);
                        partitionSize++;
                    }
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    pairs.add(new Tuple2<>("partitionSize", partitionSize));
                    return pairs.iterator();
                })
                .groupByKey()     // <-- REDUCE PHASE (R2)
                .mapToPair((pair) -> {
                    if (pair._1().equals("partitionSize")) {
                        long max = 0;
                        for (long partitionSize: pair._2())
                            max = Math.max(partitionSize, max);
                        return new Tuple2<>("maxPartitionSize", max);
                    }
                    else {
                        long sum = 0;
                        for (long c : pair._2()) {
                            sum += c;
                        }
                        return new Tuple2<>(pair._1(), sum);
                    }
                });

        Tuple2<String, Long> mostFrequent = count
                .filter((pair) -> !pair._1().equals("maxPartitionSize"))
                .reduce((x, y) -> {
                    if (x._2() > y._2())
                        return x;
                    else if (x._2() < y._2())
                        return y;
                    // ties must be broken in favor of the smaller class in alphabetical order
                    else if (x._1().compareTo(y._1()) < 0)
                        return x;
                    else
                        return y;
                });

        Long maxPartitionSize = count
                .filter((pair) -> pair._1().equals("maxPartitionSize"))
                .map(Tuple2::_2)
                .reduce((x, y) -> x); // there is only one pair with key maxPartitionSize


        System.out.println("VERSION WITH SPARK PARTITIONS");
        System.out.println("Most frequent class = " + mostFrequent);
        System.out.println("Max partition size = " + maxPartitionSize);
    }
}