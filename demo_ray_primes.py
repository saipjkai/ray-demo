import os
import time
import ray


def is_prime(num):
    result = True
    for i in range(2, num):
        if num%i == 0:
            result = False
            break
    return result


@ray.remote
def get_primes_distributed(sequence_size=100):
    primes = [2]
    for num in range(3, sequence_size+1):
        result = is_prime(num)
        if result:
            primes.append(num)
    return primes


def get_primes_local(sequence_size=100):
    primes = [2]
    for i in range(3, sequence_size+1):
        if is_prime(i):
            primes.append(i)
    return primes


# main
def main():
    sequence_size = 100000

    print("Running Local:\n")
    start = time.time()
    local_results = get_primes_local(sequence_size)
    duration = time.time() - start
    print("Sequence size: {}, execution time: {}\n".format(sequence_size, duration))

    print("Running Distributed:\n") 
    ray.init(address='auto') # ray start --head --port=6379
    start = time.time()
    distributed_results = ray.get(get_primes_distributed.remote(sequence_size))
    duration = time.time() - start
    print("Sequence size: {}, execution time: {}\n".format(sequence_size, duration))
    ray.shutdown()    

    assert local_results == distributed_results


if __name__ == "__main__":
    main()