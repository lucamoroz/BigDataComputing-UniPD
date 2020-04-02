import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

public class ClassCount {

    public static void main(String... args) {

        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        SparkConf conf = new SparkConf(true).setAppName("Homework1 - ClassCount");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        int nPartitions = Integer.parseInt(args[0]);
        JavaRDD<String> lines = sc.textFile(args[1]);
        long N = lines.count();

        JavaPairRDD<String, Long> genreCount;

        // VERSION WITH DETERMINISTIC PARTITIONS

        genreCount = lines
                .mapToPair((line) -> {
                    long id = Long.parseLong(line.split(" ")[0]);
                    String genre = line.split(" ")[1];

                    // map id into range 0..N^1/2
                    id = id % nPartitions;
                    return new Tuple2<>(id, genre);
                })
                .groupByKey()
                .flatMapToPair((pair) -> {
                    HashMap<String, Long> genreOcc = new HashMap<>();
                    for(String genre : pair._2())
                        genreOcc.put(genre, 1 + genreOcc.getOrDefault(genre, 0L));

                    return getTupleIterator(genreOcc);
                })
                .reduceByKey(Long::sum);

        // collect is required, otherwise keys are sorted locally within each task

        String sPairs = genreCount.sortByKey().collect().stream()
                .map((pair) -> "(" + pair._1() + "," + pair._2().toString() + ")")
                .collect(Collectors.joining(" "));

        System.out.println("VERSION WITH DETERMINISTIC PARTITIONS");
        System.out.println("Output pairs = " + sPairs);

        // VERSION WITH SPARK PARTITIONS
        lines = lines.repartition(nPartitions);

        genreCount = lines
                .mapPartitionsToPair((part) -> {
                    long partitionSize = 0;

                    // discard ids and collect movie genres
                    ArrayList<String> genres = new ArrayList<>();
                    while (part.hasNext()) {
                        genres.add(part.next().split(" ")[1]);
                        partitionSize++;
                    }

                    // count occurrences of genres within the partition
                    HashMap<String, Long> genreOcc = new HashMap<>();
                    for(String genre : genres)
                        genreOcc.put(genre, 1 + genreOcc.getOrDefault(genre, 0L));

                    genreOcc.put("maxPartitionSize", partitionSize); // to find biggest partition

                    return getTupleIterator(genreOcc);
                });


        System.out.println("VERSION WITH SPARK PARTITIONS");

        // print most frequent class
        genreCount.filter((p) -> !p._1().equals("maxPartitionSize"))
                .reduceByKey(Long::sum)
                .collect() // at most N_GENRES pairs
                .stream()
                .sorted(Comparator.comparing(Tuple2::_1)) // favor the smaller class in alphabetical order
                .max(Comparator.comparingLong(Tuple2::_2))
                .ifPresentOrElse(
                        (mostFreqPair) -> {
                            System.out.println("Most frequent frequent class = "
                                    + "(" + mostFreqPair._1() + "," + mostFreqPair._2() + ")");
                        },
                        () -> System.out.println("Empty RDD")
                );

        // print size of biggest partition
        genreCount.filter((p) -> p._1().equals("maxPartitionSize"))
                .collect() // at most nPartitions pairs
                .stream()
                .max((Comparator.comparingLong(Tuple2::_2)))
                .ifPresentOrElse(
                        (tuple) -> System.out.println("Max partition size: " + tuple._2()),
                        () -> System.out.println("Empty RDD")
                );
    }

    private static Iterator<Tuple2<String, Long>> getTupleIterator(HashMap<String, Long> genreOcc) {
        ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
        for (Map.Entry<String, Long> e : genreOcc.entrySet()) {
            pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
        }
        return pairs.iterator();
    }

}
