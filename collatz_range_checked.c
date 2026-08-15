/*
 * A bounded Collatz range checker, not a proof of the Collatz conjecture.
 *
 * Every integer in the inclusive range is checked exactly once.  Arithmetic
 * is rejected before 3n+1 could overflow uint64_t, and all control flow is
 * iterative.  A successful exit establishes only the finite statement
 * printed in the JSON record.
 */
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum result { CONVERGED, STEP_LIMIT, OVERFLOW };

static void usage(const char *program)
{
    fprintf(stderr, "usage: %s START END [MAX_STEPS]\n", program);
}

static int parse_positive(const char *text, uint64_t *value)
{
    char *end = NULL;
    const unsigned char *cursor = (const unsigned char *)text;
    uintmax_t parsed;

    if (*cursor == '\0')
        return 0;
    for (; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9')
            return 0;
    }
    errno = 0;
    parsed = strtoumax(text, &end, 10);
    if (errno == ERANGE || *end != '\0' || parsed == 0 || parsed > UINT64_MAX)
        return 0;
    *value = (uint64_t)parsed;
    return 1;
}

static enum result check_trajectory(uint64_t start, uint64_t max_steps,
                                    uint64_t *steps, uint64_t *last)
{
    uint64_t n = start;

    *steps = 0;
    while (n != 1) {
        if (*steps == max_steps) {
            *last = n;
            return STEP_LIMIT;
        }
        if ((n & 1U) == 0) {
            n /= 2;
        } else {
            if (n > (UINT64_MAX - 1) / 3) {
                *last = n;
                return OVERFLOW;
            }
            n = 3 * n + 1;
        }
        ++*steps;
    }
    *last = n;
    return CONVERGED;
}

int main(int argc, char **argv)
{
    uint64_t start, end, max_steps = 10000000;
    uint64_t checked = 0, maximum_steps = 0;

    if ((argc != 3 && argc != 4) ||
        !parse_positive(argv[1], &start) ||
        !parse_positive(argv[2], &end) ||
        (argc == 4 && !parse_positive(argv[3], &max_steps)) || start > end) {
        usage(argv[0]);
        return 64;
    }

    for (uint64_t n = start;; ++n) {
        uint64_t steps, last;
        enum result result = check_trajectory(n, max_steps, &steps, &last);

        if (result != CONVERGED) {
            printf("{\"status\":\"%s\",\"start\":%" PRIu64
                   ",\"end\":%" PRIu64 ",\"checked\":%" PRIu64
                   ",\"failed_input\":%" PRIu64 ",\"last_value\":%" PRIu64
                   ",\"steps\":%" PRIu64 ",\"complete\":false}\n",
                   result == OVERFLOW ? "overflow" : "step_limit",
                   start, end, checked, n, last, steps);
            return result == OVERFLOW ? 3 : 2;
        }
        ++checked;
        if (steps > maximum_steps)
            maximum_steps = steps;
        if (n == end)
            break;
    }

    printf("{\"status\":\"verified_bounded_range\",\"start\":%" PRIu64
           ",\"end\":%" PRIu64 ",\"checked\":%" PRIu64
           ",\"maximum_steps\":%" PRIu64
           ",\"complete\":true,\"claim\":\"every input in the inclusive "
           "range reached 1 using uint64 arithmetic\"}\n",
           start, end, checked, maximum_steps);
    return 0;
}
