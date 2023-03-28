import time
import ray


def fibonacci_local(sequence_size):
    fibonacci_sequence = []
    for i in range(0, sequence_size):
        if i < 2:
            fibonacci_sequence.append(i)
            continue
        fibonacci_sequence.append(fibonacci_sequence[i-1]+fibonacci_sequence[i-2])
    return fibonacci_sequence


# ray task
@ray.remote
def fibonacci_distributed(sequence_size):
    fibonacci_sequence = []
    for i in range(0, sequence_size):
        if i < 2:
            fibonacci_sequence.append(i)
            continue
        fibonacci_sequence.append(fibonacci_sequence[i-1]+fibonacci_sequence[i-2])
    return fibonacci_sequence


# main
def main():
    sequence_size = 100000
    its=8

    print("Running Local:\n")
    start = time.time()
    local_results = [fibonacci_local(sequence_size) for _ in range(its)]
    duration = time.time() - start
    print("Sequence size: {}, execution time: {}\n".format(sequence_size, duration))

    print("Running Distributed:\n")
    ray.init(address='auto') # ray start --head --port=6379
    start = time.time()
    distributed_results = ray.get([fibonacci_distributed.remote(sequence_size) for _ in range(its)])
    duration = time.time() - start
    print("Sequence size: {}, execution time: {}\n".format(sequence_size, duration))
    ray.shutdown()

    assert local_results == distributed_results


if __name__ == "__main__":
    main()
