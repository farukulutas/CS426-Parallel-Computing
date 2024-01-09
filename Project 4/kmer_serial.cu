#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util.h"

int countSubstring(const char* str, const char* sub, int* locations) {
    int count = 0;
    const char* tmp = str;
    while((tmp = strstr(tmp, sub)) != NULL) {
        locations[count++] = tmp - str;
        tmp++;
    }
    return count;
}

int main(int argc, char** argv) {
    if(argc != 5) {
        printf("Wrong arguments usage: ./kmer_serial [REFERENCE_FILE] [READ_FILE] [k] [OUTPUT_FILE]\n" );
        return 1;
    }

    char *reference_str = (char*) malloc(MAX_REF_LENGTH * sizeof(char));
    char *read_str = (char*) malloc(MAX_READ_LENGTH * sizeof(char));
    char *reference_filename = argv[1];
    char *read_filename = argv[2];
    int k = atoi(argv[3]);
    char *output_filename = argv[4];

    // Read reference string
    FILE *fp = fopen(reference_filename, "r");
    if (fp == NULL) {
        printf("Could not open file %s!\n", reference_filename);
        return 1;
    }
    if (fgets(reference_str, MAX_REF_LENGTH, fp) == NULL) {
        printf("Problem in file format!\n");
        return 1;
    }
    substring(reference_str, 0, strlen(reference_str)-1);
    fclose(fp);

    // Read queries
    StringList queries;
    initStringList(&queries, 3);

    if (read_file(read_filename, &queries) != 0) {
        printf("Error reading read file!\n");
        return 1;
    }

    FILE *output_file = fopen(output_filename, "w");
    if (output_file == NULL) {
        printf("Could not open file %s!\n", output_filename);
        return 1;
    }

    // Process each read string
    for (int i = 0; i < queries.used; i++) {
        int total_count = 0;
        int locations[MAX_REF_LENGTH];
        for (int start = 0; start <= strlen(queries.array[i]) - k; start++) {
            substring(read_str, queries.array[i], start, start + k);
            int count = countSubstring(reference_str, read_str, locations);
            total_count += count;
        }
        fprintf(output_file, "%d\n", total_count);
    }

    // Cleanup
    fclose(output_file);
    freeStringList(&queries);
    free(reference_str);
    free(read_str);

    return 0;
}
