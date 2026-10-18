#ifndef RING_BUFFER_1D_H
#define RING_BUFFER_1D_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/**
 * @file RingBuffer1D.h
 * @brief Zero-Allocation Circular FIFO Buffer for 1D Time-Series Ingestion
 *
 * Implements O(1) sample insertion and fast two-segment memcpy unwrapping
 * for sliding-window inference kernels.
 */
template <size_t WINDOW_SIZE>
class RingBuffer1D {
private:
    int8_t  buffer[WINDOW_SIZE];
    size_t  head;
    size_t  count;
    size_t  stride_counter;

public:
    RingBuffer1D() : head(0), count(0), stride_counter(0) {
        memset(buffer, 0, sizeof(buffer));
    }

    /**
     * @brief Pushes a single INT8 sensor sample into the circular FIFO (O(1)).
     */
    inline void push(int8_t sample) {
        buffer[head] = sample;
        head = (head + 1) % WINDOW_SIZE;
        if (count < WINDOW_SIZE) {
            count++;
        }
        stride_counter++;
    }

    /**
     * @brief Returns true if a full window is accumulated and stride threshold is reached.
     */
    inline bool is_ready(size_t stride) const {
        return (count >= WINDOW_SIZE) && (stride_counter >= stride);
    }

    /**
     * @brief Unwraps the circular history into a contiguous linear destination array.
     * Uses two direct memory copies rather than per-element iterations.
     */
    inline void extract_window(int8_t* destination) {
        size_t tail = head; // In a full ring buffer, head points to the oldest sample
        size_t first_segment_len = WINDOW_SIZE - tail;

        memcpy(destination, &buffer[tail], first_segment_len);

        if (tail > 0) {
            memcpy(destination + first_segment_len, &buffer[0], tail);
        }

        stride_counter = 0;
    }

    inline void reset() {
        head = 0;
        count = 0;
        stride_counter = 0;
        memset(buffer, 0, sizeof(buffer));
    }

    inline size_t size() const {
        return count;
    }
};

#endif // RING_BUFFER_1D_H