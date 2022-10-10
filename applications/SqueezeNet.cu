#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/time.h>
#include <time.h>

using namespace std;

extern "C" void computeGold(float *, const float *, const float *, unsigned int, unsigned int, unsigned int);
void NeuralNetwork();

unsigned cpu_offset;
unsigned NUM;
bool verbose_flag = true;

double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
    NUM = 1;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-') { // flag
            char flag = argv[i][1];
            switch (flag) {
            case 'n': // number of images
                i++;
                NUM = atoi(argv[i]);
                break;
            case 'o': // cpu_offset
                i++;
                cpu_offset = atoi(argv[i]);
                break;
            case 'v':
                verbose_flag = true;
                break;
            }
        }
    }

    if (!NUM) {
        printf("Usage: ./NN -n <NUM> -o <CPU offset> -v\n");
        printf("where NUM is the number of images "
               "to process in parallel (up to 10000 "
               "for the t10k-images-idx3-ubyte "
               "database file) and -v is used to "
               "display approximately what each "
               "image looks like.\n");
        return 1;
    }
    NeuralNetwork();
}

void LoadInput(int *Data_Layer_CPU) {
    FILE *fp = fopen("../data/SqueezeNet/cat.txt", "rb");
    size_t len;
    char delim[1];
    delim[0] = '\n';
    int count = 0;
    char *token;
    char *line = NULL;
    if (fp != NULL) {
        while ((getline(&line, &len, fp)) != -1) {
            token = strtok(line, delim);
            Data_Layer_CPU[count] = atof(token);
            count++;
        }
        fclose(fp);
    } else {
        printf(" File NOt FOUND\n");
    }
}

void ConvertInput(int *Data_Layer_CPU_R, int *Data_Layer_CPU_G, int *Data_Layer_CPU_B, int *Data_Layer_CPU) {
    for (int i = 0; i < 227 * 227 * 3; i += 3) {
        Data_Layer_CPU_R[i / 3] = Data_Layer_CPU[i];
        Data_Layer_CPU_G[i / 3] = Data_Layer_CPU[i + 1];
        Data_Layer_CPU_B[i / 3] = Data_Layer_CPU[i + 2];
    }
}

void InitHostMem(double *Layer1_Weights_CPU, double *fire2squeeze1x1_Weights_CPU, double *fire2expand1x1_Weights_CPU,
                 double *fire2expand3x3_Weights_CPU, double *fire3squeeze1x1_Weights_CPU,
                 double *fire3expand1x1_Weights_CPU, double *fire3expand3x3_Weights_CPU,
                 double *fire4squeeze1x1_Weights_CPU, double *fire4expand1x1_Weights_CPU,
                 double *fire4expand3x3_Weights_CPU, double *fire5squeeze1x1_Weights_CPU,
                 double *fire5expand1x1_Weights_CPU, double *fire5expand3x3_Weights_CPU,
                 double *fire6squeeze1x1_Weights_CPU, double *fire6expand1x1_Weights_CPU,
                 double *fire6expand3x3_Weights_CPU, double *fire7squeeze1x1_Weights_CPU,
                 double *fire7expand1x1_Weights_CPU, double *fire7expand3x3_Weights_CPU,
                 double *fire8squeeze1x1_Weights_CPU, double *fire8expand1x1_Weights_CPU,
                 double *fire8expand3x3_Weights_CPU, double *fire9squeeze1x1_Weights_CPU,
                 double *fire9expand1x1_Weights_CPU, double *fire9expand3x3_Weights_CPU, double *Layer10_Weights_CPU) {
    FILE *pFile1 = fopen("../data/SqueezeNet/conv1_s.txt", "rb");
    if (pFile1 != NULL) {
        char s[1000000] = "";
        fread(s, sizeof(s), 1, pFile1);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            Layer1_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 14112) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
        fclose(pFile1);
    }

    if (!pFile1) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile2 = fopen("../data/SqueezeNet/fire2_squeeze1x1.txt", "rb");
    if (pFile2 != NULL) {
        // printf("File2 Opened\n");
        char s[1000000] = "";
        fread(s, sizeof(s), 1, pFile2);
        // printf("Done2\n");
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire2squeeze1x1_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 1536) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile2) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile3 = fopen("../data/SqueezeNet/fire2_expand1x1.txt", "rb");
    if (pFile3 != NULL) {
        char s[1000000] = "";
        fread(s, sizeof(s), 1, pFile3);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire2expand1x1_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 1024) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile3) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile4 = fopen("../data/SqueezeNet/fire2_expand3x3.txt", "rb");
    if (pFile4 != NULL) {
        char s[1000000] = "";
        fread(s, sizeof(s), 1, pFile4);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire2expand3x3_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 9216) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile4) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile5 = fopen("../data/SqueezeNet/fire3_squeeze1x1.txt", "rb");
    if (pFile5 != NULL) {
        char s[1000000] = "";
        fread(s, sizeof(s), 1, pFile5);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire3squeeze1x1_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 2048) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile5) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile6 = fopen("../data/SqueezeNet/fire3_expand1x1.txt", "rb");
    if (pFile6 != NULL) {
        char s[1000000] = "";
        fread(s, sizeof(s), 1, pFile6);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire3expand1x1_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 1024) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile6) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile7 = fopen("../data/SqueezeNet/fire3_expand3x3.txt", "rb");
    if (pFile7 != NULL) {
        // printf("File7 Opened\n");
        char s[1000000] = "";
        fread(s, sizeof(s), 1, pFile7);
        // printf("Done4 %s\n",s);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire3expand3x3_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 9216) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile7) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile8 = fopen("../data/SqueezeNet/fire4_squeeze1x1.txt", "rb");
    if (pFile8 != NULL) {
        char s[1000000] = "";
        fread(s, sizeof(s), 1, pFile8);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire4squeeze1x1_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 4096) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile8) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile9 = fopen("../data/SqueezeNet/fire4_expand1x1.txt", "rb");
    if (pFile9 != NULL) {
        char s[1000000] = "";
        fread(s, sizeof(s), 1, pFile9);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire4expand1x1_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 4096) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile9) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile10 = fopen("../data/SqueezeNet/fire4_expand3x3.txt", "rb");
    if (pFile10 != NULL) {
        char s[1000000] = "";
        fread(s, sizeof(s), 1, pFile10);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire4expand3x3_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 36864) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile10) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile11 = fopen("../data/SqueezeNet/fire5_squeeze1x1.txt", "rb");
    if (pFile11 != NULL) {
        char s[1000000] = "";
        fread(s, sizeof(s), 1, pFile11);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire5squeeze1x1_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 8192) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile11) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile12 = fopen("../data/SqueezeNet/fire5_expand1x1.txt", "rb");
    if (pFile12 != NULL) {
        char s[1000000] = "";
        fread(s, sizeof(s), 1, pFile12);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire5expand1x1_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 4096) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile12) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile13 = fopen("../data/SqueezeNet/fire5_expand3x3.txt", "rb");
    if (pFile13 != NULL) {
        char s[1000000] = "";
        fread(s, sizeof(s), 1, pFile13);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire5expand3x3_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 36864) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile13) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile14 = fopen("../data/SqueezeNet/fire6_squeeze1x1.txt", "rb");
    if (pFile14 != NULL) {
        char s[1000000] = "";
        fread(s, sizeof(s), 1, pFile14);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire6squeeze1x1_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 12288) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile14) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile15 = fopen("../data/SqueezeNet/fire6_expand1x1.txt", "rb");
    if (pFile15 != NULL) {
        char s[1000000] = "";
        fread(s, sizeof(s), 1, pFile15);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire6expand1x1_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 9216) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile15) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile16 = fopen("../data/SqueezeNet/fire6_expand3x3.txt", "rb");
    if (pFile16 != NULL) {
        char s[3000000] = "";
        fread(s, sizeof(s), 1, pFile16);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire6expand3x3_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 82944) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile16) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile17 = fopen("../data/SqueezeNet/fire7_squeeze1x1.txt", "rb");
    if (pFile17 != NULL) {
        char s[1000000] = "";
        fread(s, sizeof(s), 1, pFile17);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire7squeeze1x1_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 18432) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile17) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile18 = fopen("../data/SqueezeNet/fire7_expand1x1.txt", "rb");
    if (pFile18 != NULL) {
        char s[1000000] = "";
        fread(s, sizeof(s), 1, pFile18);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire7expand1x1_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 9216) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile18) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile19 = fopen("../data/SqueezeNet/fire7_expand3x3.txt", "rb");
    if (pFile19 != NULL) {
        char s[3000000] = "";
        fread(s, sizeof(s), 1, pFile19);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire7expand3x3_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 82944) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile19) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile20 = fopen("../data/SqueezeNet/fire8_squeeze1x1.txt", "rb");
    if (pFile20 != NULL) {
        char s[3000000] = "";
        fread(s, sizeof(s), 1, pFile20);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire8squeeze1x1_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 24640) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile20) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile21 = fopen("../data/SqueezeNet/fire8_expand1x1.txt", "rb");
    if (pFile21 != NULL) {
        char s[1000000] = "";
        fread(s, sizeof(s), 1, pFile21);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire8expand1x1_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 16384) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile21) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile22 = fopen("../data/SqueezeNet/fire8_expand3x3.txt", "rb");
    if (pFile22 != NULL) {
        char s[6000000] = "";
        fread(s, sizeof(s), 1, pFile22);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire8expand3x3_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 147456) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile22) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile23 = fopen("../data/SqueezeNet/fire9_squeeze1x1.txt", "rb");
    if (pFile23 != NULL) {
        char s[3000000] = "";
        fread(s, sizeof(s), 1, pFile23);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire9squeeze1x1_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 32768) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile23) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile24 = fopen("../data/SqueezeNet/fire9_expand1x1.txt", "rb");
    if (pFile24 != NULL) {
        char s[1000000] = "";
        fread(s, sizeof(s), 1, pFile24);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire9expand1x1_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 16384) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile24) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }
    FILE *pFile25 = fopen("../data/SqueezeNet/fire9_expand3x3.txt", "rb");
    if (pFile25 != NULL) {
        char s[6000000] = "";
        fread(s, sizeof(s), 1, pFile25);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            fire9expand3x3_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 147456) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
    }

    if (!pFile25) {
        printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
        exit(1);
    }

    FILE *pFile26 = fopen("../data/SqueezeNet/conv10_s_p1.txt", "rb");
    if (pFile26 != NULL) {
        char s[8200000] = "";
        fread(s, sizeof(s), 1, pFile26);
        long int index = 0, i = 0;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            Layer10_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 256000) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
        fclose(pFile26);
    }
    FILE *pFile27 = fopen("../data/SqueezeNet/conv10_s_p2.txt", "rb");
    if (pFile27 != NULL) {
        char s[8200000] = "";
        fread(s, sizeof(s), 1, pFile27);
        long int index = 0, i = 256000;
        char delim[2];
        delim[0] = '\n';
        delim[1] = 0;
        char *temp_string = strtok(s, delim);
        while (temp_string != NULL) {
            double temp_num = atof(temp_string);
            Layer10_Weights_CPU[i] = temp_num;
            i++;
            index++;
            if (i == 512000) {
                break;
            }
            temp_string = strtok(NULL, delim);
        }
        fclose(pFile27);
    }
}

__global__ void ExecuteFirstLayer(double *Layer1_Weights_CPU, int *Data_Layer_CPU_R, int *Data_Layer_CPU_G,
                                  int *Data_Layer_CPU_B, double *Layer1_Features) {
    int x = (threadIdx.x) * 2 + 3;
    int y = (blockIdx.x) * 2 + 3;
    for (int f = 0; f < 96; f++) {
        double result = 0;
        for (int i = x - 3; i <= x + 3; i++) {
            for (int j = y - 3; j <= y + 3; j++) {
                int x_index = i - x + 3;
                int y_index = j - y + 3;
                int m = (y_index) + (x_index)*7;
                if (i < 0 || j < 0) {
                    result += 0;
                } else if (j > 226 || i > 226) {
                    result += 0;
                } else {
                    double temp = Data_Layer_CPU_R[(y_index - 3) + x * 227 + y + (x_index - 3) * 227] *
                                      Layer1_Weights_CPU[m + f * 147] +
                                  Data_Layer_CPU_G[(y_index - 3) + x * 227 + y + (x_index - 3) * 227] *
                                      Layer1_Weights_CPU[m + 49 + f * 147] +
                                  Data_Layer_CPU_B[(y_index - 3) + x * 227 + y + (x_index - 3) * 227] *
                                      Layer1_Weights_CPU[m + 98 + f * 147];
                    result += temp;
                }
            }
        }
        if (result < 0)
            result = 0;
        Layer1_Features[f * 111 * 111 + ((x - 3) / 2) * 111 + ((y - 3) / 2)] = result;
    }
}

void ExecuteFirstLayer_CPU(int gpu_block_x, double *Layer1_Weights_CPU, int *Data_Layer_CPU_R, int *Data_Layer_CPU_G,
                           int *Data_Layer_CPU_B, double *Layer1_Features) {
#pragma omp parallel for
    for (int threadIdx = 0; threadIdx < 111; threadIdx++) {
        int x = (threadIdx)*2 + 3;
        for (int blockIdx = gpu_block_x; blockIdx < 111; blockIdx++) {
            int y = (blockIdx)*2 + 3;
            for (int f = 0; f < 96; f++) {
                double result = 0;
                for (int i = x - 3; i <= x + 3; i++) {
                    for (int j = y - 3; j <= y + 3; j++) {
                        int x_index = i - x + 3;
                        int y_index = j - y + 3;
                        int m = (y_index) + (x_index)*7;
                        double temp = Data_Layer_CPU_R[(y_index - 3) + x * 227 + y + (x_index - 3) * 227] *
                                          Layer1_Weights_CPU[m + f * 147] +
                                      Data_Layer_CPU_G[(y_index - 3) + x * 227 + y + (x_index - 3) * 227] *
                                          Layer1_Weights_CPU[m + 49 + f * 147] +
                                      Data_Layer_CPU_B[(y_index - 3) + x * 227 + y + (x_index - 3) * 227] *
                                          Layer1_Weights_CPU[m + 98 + f * 147];
                        result += temp;
                    }
                }
                if (result < 0)
                    result = 0;
                Layer1_Features[f * 111 * 111 + ((x - 3) / 2) * 111 + ((y - 3) / 2)] = result;
            }
        }
    }
}

__global__ void pooling1(double *Layer2_Neurons_GPU, double *Layer2_pool_GPU) {
    int row = threadIdx.x;
    int col = blockIdx.x;
    double max = 0.0;

    for (int output = 0; output < 96; output++) {
        if (row % 2 != 0) {
            if (col % 2 != 0) {
                for (int i = row - 1; i <= row + 1; i++) {
                    if (i > 110)
                        break;
                    for (int j = col - 1; j <= col + 1; j++) {
                        if (j > 110)
                            break;
                        if (max < ((Layer2_Neurons_GPU[output * 111 * 111 + i * 111 + j])))
                            max = ((Layer2_Neurons_GPU[output * 111 * 111 + i * 111 + j]));
                    }
                }
                Layer2_pool_GPU[output * 55 * 55 + ((row - 1) / 2) * 55 + (col - 1) / 2] = max;
                max = 0.0;
            }
        }
    }

    __syncthreads();
}

// kernel 2
void pooling1_CPU(int gpu_block_x, double *Layer2_Neurons_GPU, double *Layer2_pool_GPU) {
    if (gpu_block_x < 1)
        gpu_block_x = 1;
#pragma omp parallel for
    for (int row = gpu_block_x; row < 111; row += 2) {
        for (int col = 1; col < 111; col += 2) {
            double max = 0.0;
            for (int output = 0; output < 96; output++) {
                for (int i = row - 1; i <= row + 1 && i <= 110; i++) {
                    for (int j = col - 1; j <= col + 1 && j <= 110; j++) {
                        if (max < ((Layer2_Neurons_GPU[output * 111 * 111 + i * 111 + j])))
                            max = ((Layer2_Neurons_GPU[output * 111 * 111 + i * 111 + j]));
                    }
                }
                Layer2_pool_GPU[output * 55 * 55 + ((row - 1) / 2) * 55 + (col - 1) / 2] = max;
                max = 0.0;
            }
        }
    }
}

__global__ void Executefire2squeeze1x1(double *fire2squeeze1x1_Weights_GPU, double *fire2squeeze1x1_Features,
                                       double *Layer2_pool_GPU, int gpu_f) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < gpu_f; f++) {
        Features = 0;
        for (int n = 0; n < 96; n++) {
            Features += Layer2_pool_GPU[n * 55 * 55 + x * 55 + y] * fire2squeeze1x1_Weights_GPU[f * 96 + n];
        }
        if (Features < 0)
            Features = 0;
        fire2squeeze1x1_Features[f * 55 * 55 + x * 55 + y] = Features;
    }
    __syncthreads();
}

void Executefire2squeeze1x1_CPU(double *fire2squeeze1x1_Weights_GPU, double *fire2squeeze1x1_Features,
                                double *Layer2_pool_GPU, int gpu_f) {
    double Features = 0;
#pragma omp parallel for
    for (int x = 0; x < 55; x++) {
        for (int y = 0; y < 55; y++) {
            for (int f = gpu_f; f < 16; f++) {
                Features = 0;
                for (int n = 0; n < 96; n++) {
                    Features += Layer2_pool_GPU[n * 55 * 55 + x * 55 + y] * fire2squeeze1x1_Weights_GPU[f * 96 + n];
                }
                if (Features < 0)
                    Features = 0;
                fire2squeeze1x1_Features[f * 55 * 55 + x * 55 + y] = Features;
            }
        }
    }
}

__global__ void Executefire2expand1x1(double *fire2expand1x1_Weights_GPU, double *fire2_Features,
                                      double *fire2squeeze1x1_Features, int gpu_f) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < gpu_f; f++) {
        Features = 0;
        for (int n = 0; n < 16; n++) {
            double result = 0;
            result = fire2squeeze1x1_Features[n * 55 * 55 + x * 55 + y] * fire2expand1x1_Weights_GPU[f * 16 + n];
            Features += result;
        }
        if (Features < 0)
            Features = 0;
        fire2_Features[f * 55 * 55 + x * 55 + y] = Features;
    }
    __syncthreads();
}

void Executefire2expand1x1_CPU(double *fire2expand1x1_Weights_GPU, double *fire2_Features,
                               double *fire2squeeze1x1_Features, int gpu_f) {
    double Features = 0;
#pragma omp parallel for
    for (int x = 0; x < 55; x++) {
        for (int y = 0; y < 55; y++) {
            for (int f = gpu_f; f < 64; f++) {
                Features = 0;
                for (int n = 0; n < 16; n++) {
                    double result = 0;
                    result =
                        fire2squeeze1x1_Features[n * 55 * 55 + x * 55 + y] * fire2expand1x1_Weights_GPU[f * 16 + n];
                    Features += result;
                }
                if (Features < 0)
                    Features = 0;
                fire2_Features[f * 55 * 55 + x * 55 + y] = Features;
            }
        }
    }
}

__global__ void Executefire2expand3x3(double *fire2expand3x3_Weights_GPU, double *fire2_Features,
                                      double *fire2squeeze1x1_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 64; f++) {
        Features = 0;
        for (int n = 0; n < 16; n++) {
            double result = 0;
            for (int i = x - 1; i <= x + 1; i++) {
                for (int j = y - 1; j <= y + 1; j++) {
                    int x_index = i - x + 1;
                    int y_index = j - y + 1;
                    int m = (y_index) + (x_index)*3;
                    if (i < 0 || j < 0) {
                        result += 0;
                    } else if (j > 54 || i > 54) {
                        result += 0;
                    } else {
                        result += fire2squeeze1x1_Features[n * 55 * 55 + i * 55 + j] *
                                  fire2expand3x3_Weights_GPU[m + f * 9 * 16 + n * 9];
                    }
                }
            }
            Features += result;
        }
        if (Features < 0)
            Features = 0;
        fire2_Features[f * 55 * 55 + x * 55 + y] = Features;
    }
}

void Executefire2expand3x3_CPU(double *fire2expand3x3_Weights_GPU, double *fire2_Features,
                               double *fire2squeeze1x1_Features, int gpu_block_x) {
    double Features = 0;

    for (int x = gpu_block_x; x < 55; x++) {
        for (int y = 0; y < 55; y++) {
#pragma omp parallel for
            for (int f = 0; f < 64; f++) {
                Features = 0;
                for (int n = 0; n < 16; n++) {
                    double result = 0;
                    for (int i = x - 1; i <= x + 1; i++) {
                        for (int j = y - 1; j <= y + 1; j++) {
                            int x_index = i - x + 1;
                            int y_index = j - y + 1;
                            int m = (y_index) + (x_index)*3;
                            if (i < 0 || j < 0) {
                                result += 0;
                            } else if (j > 54 || i > 54) {
                                result += 0;
                            } else {
                                result += fire2squeeze1x1_Features[n * 55 * 55 + i * 55 + j] *
                                          fire2expand3x3_Weights_GPU[m + f * 9 * 16 + n * 9];
                            }
                        }
                    }
                    Features += result;
                }
                // ReLU activation function computation
                if (Features < 0)
                    Features = 0;
                fire2_Features[f * 55 * 55 + x * 55 + y] = Features;
            }
        }
    }
}

__global__ void Executefire3squeeze1x1(double *fire3squeeze1x1_Weights_GPU, double *fire3squeeze1x1_Features,
                                       double *fire2_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 16; f++) {
        Features = 0;
        for (int n = 0; n < 128; n++) {
            Features += fire2_Features[n * 55 * 55 + x * 55 + y] * fire3squeeze1x1_Weights_GPU[f * 128 + n];
        }
        if (Features < 0)
            Features = 0;
        fire3squeeze1x1_Features[f * 55 * 55 + x * 55 + y] = Features;
    }
    __syncthreads();
}

void Executefire3squeeze1x1_CPU(double *fire3squeeze1x1_Weights_GPU, double *fire3squeeze1x1_Features,
                                double *fire2_Features, int gpu_block_x) {
    double Features = 0;
#pragma omp parallel for
    for (int x = gpu_block_x; x < 55; x++) {
        for (int y = 0; y < 55; y++) {
            for (int f = 0; f < 16; f++) {
                Features = 0;
#pragma omp simd
                for (int n = 0; n < 128; n++) {
                    Features += fire2_Features[n * 55 * 55 + x * 55 + y] * fire3squeeze1x1_Weights_GPU[f * 128 + n];
                }
                // ReLU activation function computation
                if (Features < 0)
                    Features = 0;
                fire3squeeze1x1_Features[f * 55 * 55 + x * 55 + y] = Features;
            }
        }
    }
}

__global__ void Executefire3expand1x1(double *fire3expand1x1_Weights_GPU, double *fire3_Features,
                                      double *fire3squeeze1x1_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 64; f++) {
        Features = 0;
        for (int n = 0; n < 16; n++) {
            double result = 0;
            result = fire3squeeze1x1_Features[n * 55 * 55 + x * 55 + y] * fire3expand1x1_Weights_GPU[f * 16 + n];
            Features += result;
        }
        if (Features < 0)
            Features = 0;
        fire3_Features[f * 55 * 55 + x * 55 + y] = Features;
    }
    __syncthreads();
}

void Executefire3expand1x1_CPU(double *fire3expand1x1_Weights_GPU, double *fire3_Features,
                               double *fire3squeeze1x1_Features, int gpu_block_x) {
    double Features = 0;
#pragma omp parallel for
    for (int x = gpu_block_x; x < 55; x++) {
        for (int y = 0; y < 55; y++) {
            for (int f = 0; f < 64; f++) {
                Features = 0;
                for (int n = 0; n < 16; n++) {
                    double result = 0;
                    result =
                        fire3squeeze1x1_Features[n * 55 * 55 + x * 55 + y] * fire3expand1x1_Weights_GPU[f * 16 + n];
                    Features += result;
                }
                if (Features < 0)
                    Features = 0;
                fire3_Features[f * 55 * 55 + x * 55 + y] = Features;
            }
        }
    }
}

__global__ void Executefire3expand3x3(double *fire3expand3x3_Weights_GPU, double *fire3_Features,
                                      double *fire3squeeze1x1_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 64; f++) {
        Features = 0;
        for (int n = 0; n < 16; n++) {
            double result = 0;
            for (int i = x - 1; i <= x + 1; i++) {
                for (int j = y - 1; j <= y + 1; j++) {
                    int x_index = i - x + 1;
                    int y_index = j - y + 1;
                    int m = (y_index) + (x_index)*3;
                    if (i < 0 || j < 0) {
                        result += 0;
                    } else if (j > 54 || i > 54) {
                        result += 0;
                    } else {
                        result += fire3squeeze1x1_Features[n * 55 * 55 + i * 55 + j] *
                                  fire3expand3x3_Weights_GPU[m + f * 9 * 16 + n * 9];
                    }
                }
            }
            Features += result;
        }
        if (Features < 0)
            Features = 0;
        fire3_Features[f * 55 * 55 + x * 55 + y] = Features;
    }
    __syncthreads();
}

void Executefire3expand3x3_CPU(double *fire3expand3x3_Weights_GPU, double *fire3_Features,
                               double *fire3squeeze1x1_Features, int gpu_block_x) {
    double Features = 0;
    for (int x = gpu_block_x; x < 55; x++) {
#pragma omp parallel for
        for (int y = 0; y < 55; y++) {
            for (int f = 0; f < 64; f++) {
                Features = 0;
                for (int n = 0; n < 16; n++) {
                    double result = 0;
                    for (int i = x - 1; i <= x + 1; i++) {
                        for (int j = y - 1; j <= y + 1; j++) {
                            int x_index = i - x + 1;
                            int y_index = j - y + 1;
                            int m = (y_index) + (x_index)*3;
                            if (i < 0 || j < 0) {
                                result += 0;
                            } else if (j > 54 || i > 54) {
                                result += 0;
                            } else {
                                result += fire3squeeze1x1_Features[n * 55 * 55 + i * 55 + j] *
                                          fire3expand3x3_Weights_GPU[m + f * 9 * 16 + n * 9];
                            }
                        }
                    }
                    Features += result;
                }
                if (Features < 0)
                    Features = 0;
                fire3_Features[f * 55 * 55 + x * 55 + y] = Features;
            }
        }
    }
}

__global__ void Executefire4squeeze1x1(double *fire4squeeze1x1_Weights_GPU, double *fire4squeeze1x1_Features,
                                       double *fire3_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 32; f++) {
        Features = 0;
        for (int n = 0; n < 128; n++) {
            Features += fire3_Features[n * 55 * 55 + x * 55 + y] * fire4squeeze1x1_Weights_GPU[f * 128 + n];
        }
        if (Features < 0)
            Features = 0;
        fire4squeeze1x1_Features[f * 55 * 55 + x * 55 + y] = Features;
    }
    __syncthreads();
}

void Executefire4squeeze1x1_CPU(double *fire4squeeze1x1_Weights_GPU, double *fire4squeeze1x1_Features,
                                double *fire3_Features, int gpu_block_x) {
    double Features = 0;
    for (int x = gpu_block_x; x < 55; x++) {
#pragma omp parallel for
        for (int y = 0; y < 55; y++) {
            for (int f = 0; f < 32; f++) {
                Features = 0;
                for (int n = 0; n < 128; n++) {
                    Features += fire3_Features[n * 55 * 55 + x * 55 + y] * fire4squeeze1x1_Weights_GPU[f * 128 + n];
                }
                if (Features < 0)
                    Features = 0;
                fire4squeeze1x1_Features[f * 55 * 55 + x * 55 + y] = Features;
            }
        }
    }
}

__global__ void Executefire4expand1x1(double *fire4expand1x1_Weights_GPU, double *fire4_Features,
                                      double *fire4squeeze1x1_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 128; f++) {
        Features = 0;
        for (int n = 0; n < 32; n++) {
            double result = 0;
            result = fire4squeeze1x1_Features[n * 55 * 55 + x * 55 + y] * fire4expand1x1_Weights_GPU[f * 32 + n];
            Features += result;
        }
        if (Features < 0)
            Features = 0;
        fire4_Features[f * 55 * 55 + x * 55 + y] = Features;
    }
    __syncthreads();
}

__global__ void Executefire4expand3x3(double *fire4expand3x3_Weights_GPU, double *fire4_Features,
                                      double *fire4squeeze1x1_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 128; f++) {
        Features = 0;
        for (int n = 0; n < 32; n++) {
            double result = 0;
            for (int i = x - 1; i <= x + 1; i++) {
                for (int j = y - 1; j <= y + 1; j++) {
                    int x_index = i - x + 1;
                    int y_index = j - y + 1;
                    int m = (y_index) + (x_index)*3;
                    if (i < 0 || j < 0) {
                        result += 0;
                    } else if (j > 54 || i > 54) {
                        result += 0;
                    } else {
                        result += fire4squeeze1x1_Features[n * 55 * 55 + i * 55 + j] *
                                  fire4expand3x3_Weights_GPU[m + f * 9 * 32 + n * 9];
                    }
                }
            }
            Features += result;
        }
        if (Features < 0)
            Features = 0;
        fire4_Features[f * 55 * 55 + x * 55 + y] = Features;
    }
    __syncthreads();
}

__global__ void pooling4(double *Layer4_Neurons_GPU, double *Layer4_pool_GPU, int out, int out_fr, int out_fc,
                         int kernel, int stride_width, int in_fr, int in_fc) {
    int row = threadIdx.x;
    int col = blockIdx.x;
    double max = 0.0;
    {
        for (int output = 0; output < 256; output++) {
            if (row % 2 != 0) {
                if (col % 2 != 0) {
                    for (int i = row - 1; i <= row + 1; i++) {
                        if (i > 54)
                            break;
                        for (int j = col - 1; j <= col + 1; j++) {
                            if (j > 54)
                                break;
                            if (max < ((Layer4_Neurons_GPU[output * 55 * 55 + i * 55 + j])))
                                max = ((Layer4_Neurons_GPU[output * 55 * 55 + i * 55 + j]));
                        }
                    }
                    Layer4_pool_GPU[output * 27 * 27 + ((row - 1) / 2) * 27 + (col - 1) / 2] = max;
                    max = 0.0;
                }
            }
        }
    }
    __syncthreads();
}

void pooling4_CPU(double *Layer4_Neurons_GPU, double *Layer4_pool_GPU, int out, int out_fr, int out_fc, int kernel,
                  int stride_width, int in_fr, int in_fc, int gpu_block_x) {
    if (gpu_block_x < 1)
        gpu_block_x = 1;

#pragma omp parallel for
    for (int row = 1; row < 55; row += 2) {
        for (int col = gpu_block_x; col < 55; col += 2) {
            double max = 0.0;
            for (int output = 0; output < 256; output++) {
                for (int i = row - 1; i <= row + 1; i++) {
                    for (int j = col - 1; j <= col + 1; j++) {
                        if (max < ((Layer4_Neurons_GPU[output * 55 * 55 + i * 55 + j])))
                            max = ((Layer4_Neurons_GPU[output * 55 * 55 + i * 55 + j]));
                    }
                }
                Layer4_pool_GPU[output * 27 * 27 + ((row - 1) / 2) * 27 + (col - 1) / 2] = max;
                max = 0.0;
            }
        }
    }
}

__global__ void Executefire5squeeze1x1(double *fire5squeeze1x1_Weights_GPU, double *fire5squeeze1x1_Features,
                                       double *fire4_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 32; f++) {
        Features = 0;
        for (int n = 0; n < 256; n++) {
            Features += fire4_Features[n * 27 * 27 + x * 27 + y] * fire5squeeze1x1_Weights_GPU[f * 256 + n];
        }
        if (Features < 0)
            Features = 0;
        fire5squeeze1x1_Features[f * 27 * 27 + x * 27 + y] = Features;
    }
    __syncthreads();
}

__global__ void Executefire5expand1x1(double *fire5expand1x1_Weights_GPU, double *fire5_Features,
                                      double *fire5squeeze1x1_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 128; f++) {
        Features = 0;
        for (int n = 0; n < 32; n++) {
            double result = 0;
            result = fire5squeeze1x1_Features[n * 27 * 27 + x * 27 + y] * fire5expand1x1_Weights_GPU[f * 32 + n];
            Features += result;
        }
        if (Features < 0)
            Features = 0;
        fire5_Features[f * 27 * 27 + x * 27 + y] = Features;
    }
    __syncthreads();
}

__global__ void Executefire5expand3x3(double *fire5expand3x3_Weights_GPU, double *fire5_Features,
                                      double *fire5squeeze1x1_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 128; f++) {
        Features = 0;
        for (int n = 0; n < 32; n++) {
            double result = 0;
            for (int i = x - 1; i <= x + 1; i++) {
                for (int j = y - 1; j <= y + 1; j++) {
                    int x_index = i - x + 1;
                    int y_index = j - y + 1;
                    int m = (y_index) + (x_index)*3;
                    if (i < 0 || j < 0) {
                        result += 0;
                    } else if (j > 26 || i > 26) {
                        result += 0;
                    } else {
                        result += fire5squeeze1x1_Features[n * 27 * 27 + i * 27 + j] *
                                  fire5expand3x3_Weights_GPU[m + f * 9 * 32 + n * 9];
                    }
                }
            }
            Features += result;
        }
        if (Features < 0)
            Features = 0;
        fire5_Features[f * 27 * 27 + x * 27 + y] = Features;
    }
    __syncthreads();
}

__global__ void Executefire6squeeze1x1(double *fire6squeeze1x1_Weights_GPU, double *fire6squeeze1x1_Features,
                                       double *fire5_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 48; f++) {
        Features = 0;
        for (int n = 0; n < 256; n++) {
            Features += fire5_Features[n * 27 * 27 + x * 27 + y] * fire6squeeze1x1_Weights_GPU[f * 256 + n];
        }
        if (Features < 0)
            Features = 0;
        fire6squeeze1x1_Features[f * 27 * 27 + x * 27 + y] = Features;
    }
    __syncthreads();
}

__global__ void Executefire6expand1x1(double *fire6expand1x1_Weights_GPU, double *fire6_Features,
                                      double *fire6squeeze1x1_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 192; f++) {
        Features = 0;
        for (int n = 0; n < 48; n++) {
            double result = 0;
            result = fire6squeeze1x1_Features[n * 27 * 27 + x * 27 + y] * fire6expand1x1_Weights_GPU[f * 48 + n];
            Features += result;
        }
        if (Features < 0)
            Features = 0;
        fire6_Features[f * 27 * 27 + x * 27 + y] = Features;
    }
    __syncthreads();
}

__global__ void Executefire6expand3x3(double *fire6expand3x3_Weights_GPU, double *fire6_Features,
                                      double *fire6squeeze1x1_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 192; f++) {
        Features = 0;
        for (int n = 0; n < 48; n++) {
            double result = 0;
            for (int i = x - 1; i <= x + 1; i++) {
                for (int j = y - 1; j <= y + 1; j++) {
                    int x_index = i - x + 1;
                    int y_index = j - y + 1;
                    int m = (y_index) + (x_index)*3;
                    if (i < 0 || j < 0) {
                        result += 0;
                    } else if (j > 26 || i > 26) {
                        result += 0;
                    } else {
                        result += fire6squeeze1x1_Features[n * 27 * 27 + i * 27 + j] *
                                  fire6expand3x3_Weights_GPU[m + f * 9 * 48 + n * 9];
                    }
                }
            }
            Features += result;
        }
        if (Features < 0)
            Features = 0;
        fire6_Features[f * 27 * 27 + x * 27 + y] = Features;
    }
    __syncthreads();
}

__global__ void Executefire7squeeze1x1(double *fire7squeeze1x1_Weights_GPU, double *fire7squeeze1x1_Features,
                                       double *fire6_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 48; f++) {
        Features = 0;
        for (int n = 0; n < 384; n++) {
            Features += fire6_Features[n * 27 * 27 + x * 27 + y] * fire7squeeze1x1_Weights_GPU[f * 384 + n];
        }
        if (Features < 0)
            Features = 0;
        fire7squeeze1x1_Features[f * 27 * 27 + x * 27 + y] = Features;
    }
    __syncthreads();
}

__global__ void Executefire7expand1x1(double *fire7expand1x1_Weights_GPU, double *fire7_Features,
                                      double *fire7squeeze1x1_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 192; f++) {
        Features = 0;
        for (int n = 0; n < 48; n++) {
            double result = 0;
            result = fire7squeeze1x1_Features[n * 27 * 27 + x * 27 + y] * fire7expand1x1_Weights_GPU[f * 48 + n];
            Features += result;
        }
        if (Features < 0)
            Features = 0;
        fire7_Features[f * 27 * 27 + x * 27 + y] = Features;
    }
    __syncthreads();
}

__global__ void Executefire7expand3x3(double *fire7expand3x3_Weights_GPU, double *fire7_Features,
                                      double *fire7squeeze1x1_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 192; f++) {
        Features = 0;
        for (int n = 0; n < 48; n++) {
            double result = 0;
            for (int i = x - 1; i <= x + 1; i++) {
                for (int j = y - 1; j <= y + 1; j++) {
                    int x_index = i - x + 1;
                    int y_index = j - y + 1;
                    int m = (y_index) + (x_index)*3;
                    if (i < 0 || j < 0) {
                        result += 0;
                    } else if (j > 26 || i > 26) {
                        result += 0;
                    } else {
                        result += fire7squeeze1x1_Features[n * 27 * 27 + i * 27 + j] *
                                  fire7expand3x3_Weights_GPU[m + f * 9 * 48 + n * 9];
                    }
                }
            }
            Features += result;
        }
        if (Features < 0)
            Features = 0;
        fire7_Features[f * 27 * 27 + x * 27 + y] = Features;
    }
    __syncthreads();
}

__global__ void Executefire8squeeze1x1(double *fire8squeeze1x1_Weights_GPU, double *fire8squeeze1x1_Features,
                                       double *fire7_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 64; f++) {
        Features = 0;
        for (int n = 0; n < 384; n++) {
            Features += fire7_Features[n * 27 * 27 + x * 27 + y] * fire8squeeze1x1_Weights_GPU[f * 384 + n];
        }
        if (Features < 0)
            Features = 0;
        fire8squeeze1x1_Features[f * 27 * 27 + x * 27 + y] = Features + fire8squeeze1x1_Weights_GPU[24576 + f];
    }
    __syncthreads();
}

__global__ void Executefire8expand1x1(double *fire8expand1x1_Weights_GPU, double *fire8_Features,
                                      double *fire8squeeze1x1_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 256; f++) {
        Features = 0;
        for (int n = 0; n < 64; n++) {
            double result = 0;
            result = fire8squeeze1x1_Features[n * 27 * 27 + x * 27 + y] * fire8expand1x1_Weights_GPU[f * 64 + n];
            Features += result;
        }
        if (Features < 0)
            Features = 0;
        fire8_Features[f * 27 * 27 + x * 27 + y] = Features;
    }
    __syncthreads();
}

__global__ void Executefire8expand3x3(double *fire8expand3x3_Weights_GPU, double *fire8_Features,
                                      double *fire8squeeze1x1_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 256; f++) {
        Features = 0;
        for (int n = 0; n < 64; n++) {
            double result = 0;
            for (int i = x - 1; i <= x + 1; i++) {
                for (int j = y - 1; j <= y + 1; j++) {
                    int x_index = i - x + 1;
                    int y_index = j - y + 1;
                    int m = (y_index) + (x_index)*3;
                    if (i < 0 || j < 0) {
                        result += 0;
                    } else if (j > 26 || i > 26) {
                        result += 0;
                    } else {
                        result += fire8squeeze1x1_Features[n * 27 * 27 + i * 27 + j] *
                                  fire8expand3x3_Weights_GPU[m + f * 9 * 64 + n * 9];
                    }
                }
            }
            Features += result;
        }
        if (Features < 0)
            Features = 0;
        fire8_Features[f * 27 * 27 + x * 27 + y] = Features;
    }
    __syncthreads();
}

__global__ void pooling8(double *Layer8_Neurons_GPU, double *Layer8_pool_GPU, int out, int out_fr, int out_fc,
                         int kernel, int stride_width, int in_fr, int in_fc) {
    int row = threadIdx.x;
    int col = blockIdx.x;
    double max = 0.0;
    {
        for (int output = 0; output < 512; output++) {
            if (row % 2 != 0) {
                if (col % 2 != 0) {
                    for (int i = row - 1; i <= row + 1; i++) {
                        if (i > 26)
                            break;
                        for (int j = col - 1; j <= col + 1; j++) {
                            if (j > 26)
                                break;
                            if (max < ((Layer8_Neurons_GPU[output * 27 * 27 + i * 27 + j])))
                                max = ((Layer8_Neurons_GPU[output * 27 * 27 + i * 27 + j]));
                        }
                    }
                    Layer8_pool_GPU[output * 13 * 13 + ((row - 1) / 2) * 13 + (col - 1) / 2] = max;
                    max = 0.0;
                }
            }
        }
    }
    __syncthreads();
}

void pooling8_CPU(double *Layer8_Neurons_GPU, double *Layer8_pool_GPU, int out, int out_fr, int out_fc, int kernel,
                  int stride_width, int in_fr, int in_fc, int gpu_block_x) {
    if (gpu_block_x < 1)
        gpu_block_x = 1;

#pragma omp parallel for
    for (int row = gpu_block_x; row < 27; row += 2) {
        for (int col = 1; col < 27; col += 2) {
            double max = 0.0;
            for (int output = 0; output < 512; output++) {
                for (int i = row - 1; i <= row + 1; i++) {
                    for (int j = col - 1; j <= col + 1; j++) {
                        if (max < ((Layer8_Neurons_GPU[output * 27 * 27 + i * 27 + j])))
                            max = ((Layer8_Neurons_GPU[output * 27 * 27 + i * 27 + j]));
                    }
                }
                Layer8_pool_GPU[output * 13 * 13 + ((row - 1) / 2) * 13 + (col - 1) / 2] = max;
                max = 0.0;
            }
        }
    }
}

__global__ void Executefire9squeeze1x1(double *fire9squeeze1x1_Weights_GPU, double *fire9squeeze1x1_Features,
                                       double *fire8_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 64; f++) {
        Features = 0;
        for (int n = 0; n < 512; n++) {
            Features += fire8_Features[n * 13 * 13 + x * 13 + y] * fire9squeeze1x1_Weights_GPU[f * 512 + n];
        }
        if (Features < 0)
            Features = 0;
        fire9squeeze1x1_Features[f * 13 * 13 + x * 13 + y] = Features; // +
                                                                       // fire8squeeze1x1_Weights_GPU[24576
                                                                       // + f];
                                                                       // printf("%.8f ",Features);
    }
    __syncthreads();
}

__global__ void Executefire9expand1x1(double *fire9expand1x1_Weights_GPU, double *fire9_Features,
                                      double *fire9squeeze1x1_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 256; f++) {
        Features = 0;
        for (int n = 0; n < 64; n++) {
            double result = 0;
            result = fire9squeeze1x1_Features[n * 13 * 13 + x * 13 + y] * fire9expand1x1_Weights_GPU[f * 64 + n];
            Features += result;
        }
        if (Features < 0)
            Features = 0;
        fire9_Features[f * 13 * 13 + x * 13 + y] = Features;
    }
    __syncthreads();
}

__global__ void Executefire9expand3x3(double *fire9expand3x3_Weights_GPU, double *fire9_Features,
                                      double *fire9squeeze1x1_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    for (int f = 0; f < 256; f++) {
        Features = 0;
        for (int n = 0; n < 64; n++) {
            double result = 0;
            for (int i = x - 1; i <= x + 1; i++) {
                for (int j = y - 1; j <= y + 1; j++) {
                    int x_index = i - x + 1;
                    int y_index = j - y + 1;
                    int m = (y_index) + (x_index)*3;
                    if (i < 0 || j < 0) {
                        result += 0;
                    } else if (j > 12 || i > 12) {
                        result += 0;
                    } else {
                        result += fire9squeeze1x1_Features[n * 13 * 13 + i * 13 + j] *
                                  fire9expand3x3_Weights_GPU[m + f * 9 * 64 + n * 9];
                    }
                }
            }
            Features += result;
        }
        if (Features < 0)
            Features = 0;
        fire9_Features[f * 13 * 13 + x * 13 + y] = Features;
    }
    __syncthreads();
}

__global__ void ExecuteTenthLayer(double *Layer10_Weights_GPU, double *fire9_Features, double *Layer10_Features) {
    double Features = 0;
    int x = threadIdx.x;
    int y = blockIdx.x;
    if (x != 0 && x != 14 && y != 0 && y != 14) {
        for (int f = 0; f < 1000; f++) {
            Features = 0;
            for (int n = 0; n < 512; n++) {
                Features += fire9_Features[n * 13 * 13 + (x - 1) * 13 + y - 1] * Layer10_Weights_GPU[f * 512 + n];
            }
            if (Features < 0)
                Features = 0;
            Layer10_Features[f * 15 * 15 + x * 15 + y] = Features;
        }
    }
    __syncthreads();
}

void ExecuteTenthLayer_CPU(double *Layer10_Weights_GPU, double *fire9_Features, double *Layer10_Features,
                           int gpu_block_x) {
    double Features = 0;
    if (gpu_block_x < 1)
        gpu_block_x = 1;

#pragma omp parallel for
    for (int f = 0; f < 1000; f++) {
        for (int x = gpu_block_x; x < 14; x++) {
            for (int y = 1; y < 14; y++) {
                Features = 0;
#pragma omp simd
                for (int n = 0; n < 512; n++) {
                    Features += fire9_Features[n * 13 * 13 + (x - 1) * 13 + y - 1] * Layer10_Weights_GPU[f * 512 + n];
                }
                if (Features < 0)
                    Features = 0;
                Layer10_Features[f * 15 * 15 + x * 15 + y] = Features;
            }
        }
    }
}

__global__ void global_pooling(double *Layer10_Features, double *output_GPU) {
    int tid = threadIdx.x;
    double avg = 0.0;
    for (int i = 0; i < 15; i++) {
        for (int j = 0; j <= 15; j++) {
            avg += Layer10_Features[tid * 225 + i * 15 + j];
        }
    }
    output_GPU[tid] = avg / 225;
    __syncthreads();
}

void global_pooling_CPU(double *Layer10_Features, double *output_GPU, int gpu_block_x) {
#pragma omp parallel for
    for (int tid = gpu_block_x; tid < 1000; tid++) {
        double avg = 0.0;
        for (int i = 0; i < 15; i++) {
            for (int j = 0; j <= 15; j++) {
                avg += Layer10_Features[tid * 225 + i * 15 + j];
            }
        }
        output_GPU[tid] = avg / 225;
    }
}

int predict_class(double *output_CPU) {
    double max = 0;
    int predicted_class = 0;
    for (int i = 0; i < 1000; i++) {
        if (output_CPU[i] > max) {
            max = output_CPU[i];
            predicted_class = i;
        }
    }
    return predicted_class;
}

void NeuralNetwork() {
    cudaError_t err;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "There is no device.\n");
        exit(EXIT_FAILURE);
    }
    int dev;
    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        if (deviceProp.major >= 1)
            break;
    }
    if (dev == deviceCount) {
        fprintf(stderr, "There is no device "
                        "supporting CUDA.\n");
        exit(EXIT_FAILURE);
    } else
        cudaSetDevice(dev);
    double *Layer1_Weights_CPU;
    double t1 = gettime();
    cudaMallocManaged((void **)&Layer1_Weights_CPU, sizeof(double) * 14112 * NUM, cudaMemAttachHost);
    int *Data_Layer_CPU_R;
    cudaMallocManaged((void **)&Data_Layer_CPU_R, 227 * 227 * NUM * sizeof(int), cudaMemAttachHost);

    int *Data_Layer_CPU_G;
    cudaMallocManaged((void **)&Data_Layer_CPU_G, 227 * 227 * NUM * sizeof(int), cudaMemAttachHost);

    int *Data_Layer_CPU_B;
    cudaMallocManaged((void **)&Data_Layer_CPU_B, 227 * 227 * NUM * sizeof(int), cudaMemAttachHost);

    int *Data_Layer_CPU = (int *)malloc(3 * 227 * 227 * NUM * sizeof(int));

    double *fire2squeeze1x1_Weights_CPU;
    cudaMallocManaged((void **)&fire2squeeze1x1_Weights_CPU, 1536 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire2expand1x1_Weights_CPU;
    cudaMallocManaged((void **)&fire2expand1x1_Weights_CPU, 1024 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire2expand3x3_Weights_CPU;
    cudaMallocManaged((void **)&fire2expand3x3_Weights_CPU, 9216 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire3squeeze1x1_Weights_CPU;
    cudaMallocManaged((void **)&fire3squeeze1x1_Weights_CPU, 2048 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire3expand1x1_Weights_CPU;
    cudaMallocManaged((void **)&fire3expand1x1_Weights_CPU, 1024 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire3expand3x3_Weights_CPU;
    cudaMallocManaged((void **)&fire3expand3x3_Weights_CPU, 9216 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire4squeeze1x1_Weights_CPU;
    cudaMallocManaged((void **)&fire4squeeze1x1_Weights_CPU, 4096 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire4expand1x1_Weights_CPU;
    cudaMallocManaged((void **)&fire4expand1x1_Weights_CPU, 4096 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire4expand3x3_Weights_CPU;
    cudaMallocManaged((void **)&fire4expand3x3_Weights_CPU, 36864 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire5squeeze1x1_Weights_CPU;
    cudaMallocManaged((void **)&fire5squeeze1x1_Weights_CPU, 8192 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire5expand1x1_Weights_CPU;
    cudaMallocManaged((void **)&fire5expand1x1_Weights_CPU, 4096 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire5expand3x3_Weights_CPU;
    cudaMallocManaged((void **)&fire5expand3x3_Weights_CPU, 36864 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire6squeeze1x1_Weights_CPU;
    cudaMallocManaged((void **)&fire6squeeze1x1_Weights_CPU, 12288 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire6expand1x1_Weights_CPU;
    cudaMallocManaged((void **)&fire6expand1x1_Weights_CPU, 9216 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire6expand3x3_Weights_CPU;
    cudaMallocManaged((void **)&fire6expand3x3_Weights_CPU, 82944 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire7squeeze1x1_Weights_CPU;
    cudaMallocManaged((void **)&fire7squeeze1x1_Weights_CPU, 18432 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire7expand1x1_Weights_CPU;
    cudaMallocManaged((void **)&fire7expand1x1_Weights_CPU, 9216 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire7expand3x3_Weights_CPU;
    cudaMallocManaged((void **)&fire7expand3x3_Weights_CPU, 82944 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire8squeeze1x1_Weights_CPU;
    cudaMallocManaged((void **)&fire8squeeze1x1_Weights_CPU, 24640 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire8expand1x1_Weights_CPU;
    cudaMallocManaged((void **)&fire8expand1x1_Weights_CPU, 16384 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire8expand3x3_Weights_CPU;
    cudaMallocManaged((void **)&fire8expand3x3_Weights_CPU, 147456 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire9squeeze1x1_Weights_CPU;
    cudaMallocManaged((void **)&fire9squeeze1x1_Weights_CPU, 32768 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire9expand1x1_Weights_CPU;
    cudaMallocManaged((void **)&fire9expand1x1_Weights_CPU, 16384 * NUM * sizeof(double), cudaMemAttachHost);

    double *fire9expand3x3_Weights_CPU;
    cudaMallocManaged((void **)&fire9expand3x3_Weights_CPU, 147456 * NUM * sizeof(double), cudaMemAttachHost);

    double *Layer10_Weights_CPU;
    cudaMallocManaged((void **)&Layer10_Weights_CPU, 512000 * NUM * sizeof(double), cudaMemAttachHost);

    double *Layer1_Features;
    cudaMallocManaged((void **)&Layer1_Features, 111 * 111 * 96 * NUM * sizeof(double), cudaMemAttachHost);

    double t2 = gettime();

    cout << cpu_offset << "\t";
    InitHostMem(Layer1_Weights_CPU, fire2squeeze1x1_Weights_CPU, fire2expand1x1_Weights_CPU, fire2expand3x3_Weights_CPU,
                fire3squeeze1x1_Weights_CPU, fire3expand1x1_Weights_CPU, fire3expand3x3_Weights_CPU,
                fire4squeeze1x1_Weights_CPU, fire4expand1x1_Weights_CPU, fire4expand3x3_Weights_CPU,
                fire5squeeze1x1_Weights_CPU, fire5expand1x1_Weights_CPU, fire5expand3x3_Weights_CPU,
                fire6squeeze1x1_Weights_CPU, fire6expand1x1_Weights_CPU, fire6expand3x3_Weights_CPU,
                fire7squeeze1x1_Weights_CPU, fire7expand1x1_Weights_CPU, fire7expand3x3_Weights_CPU,
                fire8squeeze1x1_Weights_CPU, fire8expand1x1_Weights_CPU, fire8expand3x3_Weights_CPU,
                fire9squeeze1x1_Weights_CPU, fire9expand1x1_Weights_CPU, fire9expand3x3_Weights_CPU,
                Layer10_Weights_CPU);
    LoadInput(Data_Layer_CPU);
    ConvertInput(Data_Layer_CPU_R, Data_Layer_CPU_G, Data_Layer_CPU_B, Data_Layer_CPU);
    t1 = gettime();
    cudaEventRecord(start);

    int cpu_block_x, gpu_block_x;
    bool corun_1 = true;
    double kernel_start_time = gettime();
    if (corun_1) {
        cpu_block_x = 111 * cpu_offset / 100;
        gpu_block_x = 111 - cpu_block_x;

        if (gpu_block_x > 0) {
            dim3 n_threads(111, 1, 1);
            dim3 n_blocks(gpu_block_x, 1, 1);
            ExecuteFirstLayer<<<n_blocks, n_threads>>>(Layer1_Weights_CPU, Data_Layer_CPU_R, Data_Layer_CPU_G,
                                                       Data_Layer_CPU_B, Layer1_Features);
        }
        if (cpu_block_x > 0) {
            ExecuteFirstLayer_CPU(gpu_block_x, Layer1_Weights_CPU, Data_Layer_CPU_R, Data_Layer_CPU_G, Data_Layer_CPU_B,
                                  Layer1_Features);
        }
        if (gpu_block_x > 0)
            cudaDeviceSynchronize();
    } else {
        dim3 n_threads(111, 1, 1);
        dim3 n_blocks(111, 1, 1);
        ExecuteFirstLayer<<<n_blocks, n_threads>>>(Layer1_Weights_CPU, Data_Layer_CPU_R, Data_Layer_CPU_G,
                                                   Data_Layer_CPU_B, Layer1_Features);
        cudaDeviceSynchronize();
    }
    t2 = gettime();
    // kernel 1

    cudaFree(Layer1_Weights_CPU);
    cudaFree(Data_Layer_CPU_R);
    cudaFree(Data_Layer_CPU_G);
    cudaFree(Data_Layer_CPU_B);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "1st LayerKernel execution failed (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    double *Pool_Layer_Features;
    err = cudaMallocManaged((void **)&Pool_Layer_Features, 290400 * NUM * sizeof(double), cudaMemAttachHost);
    bool corun_2 = true;
    t1 = gettime();
    if (corun_2) {
        cpu_block_x = 111 * cpu_offset / 100;
        gpu_block_x = 111 - cpu_block_x;
        if (gpu_block_x > 0) {
            if (gpu_block_x % 2 == 0)
                gpu_block_x--;
            dim3 n_threads_pool(gpu_block_x, 1, 1);
            dim3 n_blocks_pool(111, 1, 1);
            pooling1<<<n_blocks_pool, n_threads_pool>>>(Layer1_Features, Pool_Layer_Features);
        }
        if (cpu_block_x > 0) {
            pooling1_CPU(gpu_block_x, Layer1_Features, Pool_Layer_Features);
        }
        if (gpu_block_x > 0)
            cudaDeviceSynchronize();
    } else {
        dim3 n_threads_pool(111, 1, 1);
        dim3 n_blocks_pool(111, 1, 1);
        pooling1<<<n_blocks_pool, n_threads_pool>>>(Layer1_Features, Pool_Layer_Features);
        cudaDeviceSynchronize();
    }

    t2 = gettime();
    // kernel 2
    // Fire
    // 2///////////////////////////////////////////////////////////////////////////////////////////////////////////
    double *fire2squeeze1x1_Features;
    err = cudaMallocManaged((void **)&fire2squeeze1x1_Features, 55 * 55 * 16 * NUM * sizeof(double), cudaMemAttachHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate device data "
                "(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    bool corun_3 = true;
    t1 = gettime();
    if (corun_3) {
        int gpu_f = 16 - 16 * cpu_offset / 100;
        if (gpu_block_x > 0) {
            dim3 n_threads1(55, 1, 1);
            dim3 n_blocks1(55, 1, 1);
            Executefire2squeeze1x1<<<n_blocks1, n_threads1>>>(fire2squeeze1x1_Weights_CPU, fire2squeeze1x1_Features,
                                                              Pool_Layer_Features, gpu_f);
        }
        if (cpu_block_x > 0) {
            Executefire2squeeze1x1_CPU(fire2squeeze1x1_Weights_CPU, fire2squeeze1x1_Features, Pool_Layer_Features,
                                       gpu_f);
        }
        if (gpu_block_x > 0)
            cudaDeviceSynchronize();
    } else {
        dim3 n_threads1(55, 1, 1);
        dim3 n_blocks1(55, 1, 1);
        Executefire2squeeze1x1<<<n_blocks1, n_threads1>>>(fire2squeeze1x1_Weights_CPU, fire2squeeze1x1_Features,
                                                          Pool_Layer_Features, 16);
        cudaDeviceSynchronize();
    }

    t2 = gettime();
    // kernel 3

    cudaFree(fire2squeeze1x1_Weights_CPU);

    double *fire2_Features;
    err = cudaMallocManaged((void **)&fire2_Features, 55 * 55 * 128 * NUM * sizeof(double), cudaMemAttachHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate device data "
                "(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    t1 = gettime();
    bool corun_4 = true;
    if (corun_4) {
        int gpu_f = 64 - 64 * cpu_offset / 100;

        if (gpu_block_x > 0) {
            dim3 n_threads1(55, 1, 1);
            dim3 n_blocks1(55, 1, 1);
            Executefire2expand1x1<<<n_blocks1, n_threads1>>>(fire2expand1x1_Weights_CPU, fire2_Features,
                                                             fire2squeeze1x1_Features, gpu_f);
        }
        if (cpu_block_x > 0)
            Executefire2expand1x1_CPU(fire2expand1x1_Weights_CPU, fire2_Features, fire2squeeze1x1_Features, gpu_f);
        if (gpu_block_x > 0)
            cudaDeviceSynchronize();
    } else {
        dim3 n_threads1(55, 1, 1);
        dim3 n_blocks1(55, 1, 1);
        Executefire2expand1x1<<<n_blocks1, n_threads1>>>(fire2expand1x1_Weights_CPU, fire2_Features,
                                                         fire2squeeze1x1_Features, 64);
        cudaDeviceSynchronize();
    }
    t2 = gettime();
    cudaFree(fire2expand1x1_Weights_CPU);

    t1 = gettime();
    bool corun_5 = true;
    if (corun_5) {
        // int cpu_offset_2 = 10;
        cpu_block_x = 55 * cpu_offset / 100;
        gpu_block_x = 55 - cpu_block_x;
        if (gpu_block_x > 0) {
            dim3 n_threads1(gpu_block_x, 1, 1);
            dim3 n_blocks1(55, 1, 1);
            Executefire2expand3x3<<<n_blocks1, n_threads1>>>(fire2expand3x3_Weights_CPU,
                                                             fire2_Features + (55 * 55 * 64), fire2squeeze1x1_Features);
        }
        if (cpu_block_x > 0)
            Executefire2expand3x3_CPU(fire2expand3x3_Weights_CPU, fire2_Features + (55 * 55 * 64),
                                      fire2squeeze1x1_Features, gpu_block_x);
        if (gpu_block_x > 0)
            cudaDeviceSynchronize();
    } else {
        dim3 n_threads1(55, 1, 1);
        dim3 n_blocks1(55, 1, 1);
        Executefire2expand3x3<<<n_blocks1, n_threads1>>>(fire2expand3x3_Weights_CPU, fire2_Features + (55 * 55 * 64),
                                                         fire2squeeze1x1_Features);
        cudaDeviceSynchronize();
    }
    t2 = gettime();

    cudaFree(fire2expand3x3_Weights_CPU);

    // Fire 3
    // //////////////////////////////////////////////////////////////////////////////////////////////////////////
    double *fire3squeeze1x1_Features;
    err = cudaMallocManaged((void **)&fire3squeeze1x1_Features, 55 * 55 * 16 * NUM * sizeof(double), cudaMemAttachHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate device data "
                "(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    t1 = gettime();
    bool corun_6 = true;
    if (corun_6) {
        cpu_block_x = 55 * cpu_offset / 100;
        gpu_block_x = 55 - cpu_block_x;
        if (gpu_block_x > 0) {
            dim3 n_threads1(55, 1, 1);
            dim3 n_blocks1(gpu_block_x, 1, 1);
            Executefire3squeeze1x1<<<n_blocks1, n_threads1>>>(fire3squeeze1x1_Weights_CPU, fire3squeeze1x1_Features,
                                                              fire2_Features);
        }
        if (cpu_block_x > 0)
            Executefire3squeeze1x1_CPU(fire3squeeze1x1_Weights_CPU, fire3squeeze1x1_Features, fire2_Features,
                                       gpu_block_x);
        if (gpu_block_x > 0)
            cudaDeviceSynchronize();
    } else {
        dim3 n_threads1(55, 1, 1);
        dim3 n_blocks1(55, 1, 1);
        Executefire3squeeze1x1<<<n_blocks1, n_threads1>>>(fire3squeeze1x1_Weights_CPU, fire3squeeze1x1_Features,
                                                          fire2_Features);
        cudaDeviceSynchronize();
    }
    t2 = gettime();

    cudaDeviceSynchronize();

    double *fire3_Features;
    err = cudaMallocManaged((void **)&fire3_Features, 55 * 55 * 128 * NUM * sizeof(double), cudaMemAttachHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate device data "
                "(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    t1 = gettime();
    bool corun_7 = true;
    if (corun_7) {
        cpu_block_x = 55 * cpu_offset / 100;
        gpu_block_x = 55 - cpu_block_x;
        if (gpu_block_x > 0) {
            dim3 n_threads1(gpu_block_x, 1, 1);
            dim3 n_blocks1(55, 1, 1);
            Executefire3expand1x1<<<n_blocks1, n_threads1>>>(fire3expand1x1_Weights_CPU, fire3_Features,
                                                             fire3squeeze1x1_Features);
        }
        if (cpu_block_x > 0)
            Executefire3expand1x1_CPU(fire3expand1x1_Weights_CPU, fire3_Features, fire3squeeze1x1_Features,
                                      gpu_block_x);
        if (gpu_block_x > 0)
            cudaDeviceSynchronize();
    } else {
        dim3 n_threads1(55, 1, 1);
        dim3 n_blocks1(55, 1, 1);
        Executefire3expand1x1<<<n_blocks1, n_threads1>>>(fire3expand1x1_Weights_CPU, fire3_Features,
                                                         fire3squeeze1x1_Features);
        cudaDeviceSynchronize();
    }
    t2 = gettime();

    cudaFree(fire3expand1x1_Weights_CPU);

    // HERE Time: 1.216 s

    t1 = gettime();
    bool corun_8 = true;
    if (corun_8) {
        cpu_block_x = 55 * cpu_offset / 100;
        gpu_block_x = 55 - cpu_block_x;
        if (gpu_block_x > 0) {
            dim3 n_threads1(gpu_block_x, 1, 1);
            dim3 n_blocks1(55, 1, 1);
            Executefire3expand3x3<<<n_blocks1, n_threads1>>>(fire3expand3x3_Weights_CPU,
                                                             fire3_Features + (55 * 55 * 64), fire3squeeze1x1_Features);
        }
        if (cpu_block_x > 0)
            Executefire3expand3x3_CPU(fire3expand3x3_Weights_CPU, fire3_Features + (55 * 55 * 64),
                                      fire3squeeze1x1_Features, gpu_block_x);
        if (gpu_block_x > 0)
            cudaDeviceSynchronize();
    } else {
        dim3 n_threads1(55, 1, 1);
        dim3 n_blocks1(55, 1, 1);
        Executefire3expand3x3<<<n_blocks1, n_threads1>>>(fire3expand3x3_Weights_CPU, fire3_Features + (55 * 55 * 64),
                                                         fire3squeeze1x1_Features);
        cudaDeviceSynchronize();
    }
    t2 = gettime();

    cudaFree(fire3expand3x3_Weights_CPU);

    // Fire 4
    // //////////////////////////////////////////////////////////////////////////////////////////////////////////
    double *fire4squeeze1x1_Features;
    err = cudaMallocManaged((void **)&fire4squeeze1x1_Features, 55 * 55 * 32 * NUM * sizeof(double), cudaMemAttachHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate device data "
                "(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    t1 = gettime();
    bool corun_9 = true;
    if (corun_9) {
        cpu_block_x = 55 * cpu_offset / 100;
        gpu_block_x = 55 - cpu_block_x;
        if (gpu_block_x > 0) {
            dim3 n_threads1(gpu_block_x, 1, 1);
            dim3 n_blocks1(55, 1, 1);
            Executefire4squeeze1x1<<<n_blocks1, n_threads1>>>(fire4squeeze1x1_Weights_CPU, fire4squeeze1x1_Features,
                                                              fire3_Features);
        }
        if (cpu_block_x > 0)
            Executefire4squeeze1x1_CPU(fire4squeeze1x1_Weights_CPU, fire4squeeze1x1_Features, fire3_Features,
                                       gpu_block_x);
        if (gpu_block_x > 0)
            cudaDeviceSynchronize();
    } else {
        dim3 n_threads1(55, 1, 1);
        dim3 n_blocks1(55, 1, 1);
        Executefire4squeeze1x1<<<n_blocks1, n_threads1>>>(fire4squeeze1x1_Weights_CPU, fire4squeeze1x1_Features,
                                                          fire3_Features);
        cudaDeviceSynchronize();
    }
    t2 = gettime();

    cudaFree(fire4squeeze1x1_Weights_CPU);

    double *fire4_Features;
    err = cudaMallocManaged((void **)&fire4_Features, 55 * 55 * 256 * NUM * sizeof(double), cudaMemAttachHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate device data "
                "(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    dim3 n_threads1(55, 1, 1);
    dim3 n_blocks1(55, 1, 1);
    t1 = gettime();
    Executefire4expand1x1<<<n_blocks1, n_threads1>>>(fire4expand1x1_Weights_CPU, fire4_Features,
                                                     fire4squeeze1x1_Features);
    cudaDeviceSynchronize();
    t2 = gettime();
    
    cudaFree(fire4expand1x1_Weights_CPU);

    Executefire4expand3x3<<<n_blocks1, n_threads1>>>(fire4expand3x3_Weights_CPU, fire4_Features + (55 * 55 * 128),
                                                     fire4squeeze1x1_Features);
    cudaDeviceSynchronize();
    cudaFree(fire4expand3x3_Weights_CPU);
    // Pool 4
    // ///////////////////////////////////////////////////////////////////////////////////////////
    double *Pool_Layer4_Features;
    err = cudaMallocManaged((void **)&Pool_Layer4_Features, 186624 * NUM * sizeof(double), cudaMemAttachHost);
    t1 = gettime();
    bool corun_10 = true;
    if (corun_10) {
        cpu_block_x = 55 * cpu_offset / 100;
        gpu_block_x = 55 - cpu_block_x;
        if (gpu_block_x > 0) {
            if (gpu_block_x % 2 == 0)
                gpu_block_x--;
            dim3 n_threads1(55, 1, 1);
            dim3 n_blocks1(gpu_block_x, 1, 1);
            pooling4<<<n_blocks1, n_threads1>>>(fire4_Features, Pool_Layer4_Features, 256, 27, 27, 3, 2, 55, 55);
        }
        if (cpu_block_x > 0)
            pooling4_CPU(fire4_Features, Pool_Layer4_Features, 256, 27, 27, 3, 2, 55, 55, gpu_block_x);
        if (gpu_block_x > 0)
            cudaDeviceSynchronize();
    } else {
        dim3 n_threads1(55, 1, 1);
        dim3 n_blocks1(55, 1, 1);
        pooling4<<<n_blocks1, n_threads1>>>(fire4_Features, Pool_Layer4_Features, 256, 27, 27, 3, 2, 55, 55);
        cudaDeviceSynchronize();
    }
    t2 = gettime();

    cudaDeviceSynchronize();

    // Fire 5
    // //////////////////////////////////////////////////////////////////////////////////////////////////////////
    dim3 n_threads2(27, 1, 1);
    dim3 n_blocks2(27, 1, 1);
    double *fire5squeeze1x1_Features;
    err = cudaMalloc((void **)&fire5squeeze1x1_Features, 27 * 27 * 32 * NUM * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate device data "
                "(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    Executefire5squeeze1x1<<<n_blocks2, n_threads2>>>(fire5squeeze1x1_Weights_CPU, fire5squeeze1x1_Features,
                                                      Pool_Layer4_Features);
    cudaDeviceSynchronize();
    cudaFree(fire5squeeze1x1_Weights_CPU);
    double *fire5_Features;
    err = cudaMalloc((void **)&fire5_Features, 27 * 27 * 256 * NUM * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate device data "
                "(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    t1 = gettime();
    Executefire5expand1x1<<<n_blocks2, n_threads2>>>(fire5expand1x1_Weights_CPU, fire5_Features,
                                                     fire5squeeze1x1_Features);
    cudaDeviceSynchronize();
    t2 = gettime();
    
    cudaFree(fire5expand1x1_Weights_CPU);

    Executefire5expand3x3<<<n_blocks2, n_threads2>>>(fire5expand3x3_Weights_CPU, fire5_Features + (27 * 27 * 128),
                                                     fire5squeeze1x1_Features);
    cudaDeviceSynchronize();

    cudaFree(fire5expand3x3_Weights_CPU);
    // Fire 6
    // //////////////////////////////////////////////////////////////////////////////////////////////////////////
    double *fire6squeeze1x1_Features;

    err = cudaMalloc((void **)&fire6squeeze1x1_Features, 27 * 27 * 48 * NUM * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate device data "
                "(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    Executefire6squeeze1x1<<<n_blocks2, n_threads2>>>(fire6squeeze1x1_Weights_CPU, fire6squeeze1x1_Features,
                                                      fire5_Features);
    cudaDeviceSynchronize();
    cudaFree(fire6squeeze1x1_Weights_CPU);
    double *fire6_Features;

    err = cudaMalloc((void **)&fire6_Features, 27 * 27 * 384 * NUM * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate device data "
                "(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    t1 = gettime();
    Executefire6expand1x1<<<n_blocks2, n_threads2>>>(fire6expand1x1_Weights_CPU, fire6_Features,
                                                     fire6squeeze1x1_Features);
    cudaDeviceSynchronize();
    t2 = gettime();
    
    cudaFree(fire6expand1x1_Weights_CPU);

    Executefire6expand3x3<<<n_blocks2, n_threads2>>>(fire6expand3x3_Weights_CPU, fire6_Features + (27 * 27 * 192),
                                                     fire6squeeze1x1_Features);
    cudaDeviceSynchronize();
    cudaFree(fire6expand3x3_Weights_CPU);
    // Fire 7
    // //////////////////////////////////////////////////////////////////////////////////////////////////////////
    double *fire7squeeze1x1_Features;

    err = cudaMalloc((void **)&fire7squeeze1x1_Features, 27 * 27 * 48 * NUM * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate device data "
                "(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    Executefire7squeeze1x1<<<n_blocks2, n_threads2>>>(fire7squeeze1x1_Weights_CPU, fire7squeeze1x1_Features,
                                                      fire6_Features);
    cudaDeviceSynchronize();
    cudaFree(fire7squeeze1x1_Weights_CPU);
    double *fire7_Features;

    err = cudaMalloc((void **)&fire7_Features, 27 * 27 * 384 * NUM * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate device data "
                "(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    t1 = gettime();
    Executefire7expand1x1<<<n_blocks2, n_threads2>>>(fire7expand1x1_Weights_CPU, fire7_Features,
                                                     fire7squeeze1x1_Features);
    cudaDeviceSynchronize();
    t2 = gettime();
    
    cudaFree(fire7expand1x1_Weights_CPU);

    Executefire7expand3x3<<<n_blocks2, n_threads2>>>(fire7expand3x3_Weights_CPU, fire7_Features + (27 * 27 * 192),
                                                     fire7squeeze1x1_Features);
    cudaDeviceSynchronize();
    cudaFree(fire7expand3x3_Weights_CPU);
    // Fire 8
    // //////////////////////////////////////////////////////////////////////////////////////////////////////////
    double *fire8squeeze1x1_Features_CPU;
    err = cudaMallocManaged((void **)&fire8squeeze1x1_Features_CPU, 27 * 27 * 64 * NUM * sizeof(double),
                            cudaMemAttachHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate device data "
                "(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    Executefire8squeeze1x1<<<n_blocks2, n_threads2>>>(fire8squeeze1x1_Weights_CPU, fire8squeeze1x1_Features_CPU,
                                                      fire7_Features);
    cudaDeviceSynchronize();
    cudaFree(fire8squeeze1x1_Weights_CPU);

    double *fire8_Features;

    err = cudaMallocManaged((void **)&fire8_Features, 27 * 27 * 512 * NUM * sizeof(double), cudaMemAttachHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate device data "
                "(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    t1 = gettime();
    Executefire8expand1x1<<<n_blocks2, n_threads2>>>(fire8expand1x1_Weights_CPU, fire8_Features,
                                                     fire8squeeze1x1_Features_CPU);
    cudaDeviceSynchronize();
    t2 = gettime();
    
    cudaFree(fire8expand1x1_Weights_CPU);

    Executefire8expand3x3<<<n_blocks2, n_threads2>>>(fire8expand3x3_Weights_CPU, fire8_Features + (27 * 27 * 256),
                                                     fire8squeeze1x1_Features_CPU);
    cudaDeviceSynchronize();
    cudaFree(fire8expand3x3_Weights_CPU);

    // Pool 8
    // ///////////////////////////////////////////////////////////////////////////////////////////
    double *Pool_Layer8_Features;
    err = cudaMallocManaged((void **)&Pool_Layer8_Features, 86528 * NUM * sizeof(double), cudaMemAttachHost);

    t1 = gettime();
    bool corun_18 = true;
    if (corun_18) {
        int cpu_offset_18 = 10;
        cpu_block_x = 27 * cpu_offset_18 / 100;
        gpu_block_x = 27 - cpu_block_x;
        if (gpu_block_x > 0) {
            if (gpu_block_x % 2 == 0)
                gpu_block_x--;
            dim3 n_threads2(gpu_block_x, 1, 1);
            dim3 n_blocks2(27, 1, 1);
            pooling8<<<n_blocks2, n_threads2>>>(fire8_Features, Pool_Layer8_Features, 512, 13, 13, 3, 2, 27, 27);
        }
        if (cpu_block_x > 0)
            pooling8_CPU(fire8_Features, Pool_Layer8_Features, 512, 13, 13, 3, 2, 27, 27, gpu_block_x);
        if (gpu_block_x > 0)
            cudaDeviceSynchronize();
    } else {
        dim3 n_threads2(27, 1, 1);
        dim3 n_blocks2(27, 1, 1);
        pooling8<<<n_blocks2, n_threads2>>>(fire8_Features, Pool_Layer8_Features, 512, 13, 13, 3, 2, 27, 27);
        cudaDeviceSynchronize();
    }
    t2 = gettime();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Pool 8 execution failed (error "
                "code %s)\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Fire 9
    // //////////////////////////////////////////////////////////////////////////////////////////////////////////
    dim3 n_threads3(13, 1, 1);
    dim3 n_blocks3(13, 1, 1);
    double *fire9squeeze1x1_Features_CPU;

    err = cudaMallocManaged((void **)&fire9squeeze1x1_Features_CPU, 13 * 13 * 64 * NUM * sizeof(double),
                            cudaMemAttachHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate device data "
                "(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    Executefire9squeeze1x1<<<n_blocks3, n_threads3>>>(fire9squeeze1x1_Weights_CPU, fire9squeeze1x1_Features_CPU,
                                                      Pool_Layer8_Features);
    cudaDeviceSynchronize();
    cudaFree(fire9squeeze1x1_Weights_CPU);

    double *fire9_Features;

    err = cudaMallocManaged((void **)&fire9_Features, 13 * 13 * 512 * NUM * sizeof(double), cudaMemAttachHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate device data "
                "(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    t1 = gettime();
    Executefire9expand1x1<<<n_blocks3, n_threads3>>>(fire9expand1x1_Weights_CPU, fire9_Features,
                                                     fire9squeeze1x1_Features_CPU);
    cudaDeviceSynchronize();
    cudaFree(fire9expand1x1_Weights_CPU);
    t2 = gettime();
    
    cudaFree(fire8expand1x1_Weights_CPU);

    Executefire9expand3x3<<<n_blocks3, n_threads3>>>(fire9expand3x3_Weights_CPU, fire9_Features + (13 * 13 * 256),
                                                     fire9squeeze1x1_Features_CPU);
    cudaDeviceSynchronize();
    cudaFree(fire9expand3x3_Weights_CPU);

    // Execute 10th layer
    // Layer/////////////////////////////////////////////////////////////////////////

    double *Layer10_Features;
    err = cudaMallocManaged((void **)&Layer10_Features, 1000 * 15 * 15 * NUM * sizeof(double), cudaMemAttachHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate device data "
                "(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    t1 = gettime();
    bool corun_19 = true;
    if (corun_19) {
        cpu_block_x = 15 * cpu_offset / 100;
        gpu_block_x = 15 - cpu_block_x;
        if (gpu_block_x > 0) {
            dim3 n_threads4(gpu_block_x, 1, 1);
            dim3 n_blocks4(15, 1, 1);
            ExecuteTenthLayer<<<n_blocks4, n_threads4>>>(Layer10_Weights_CPU, fire9_Features, Layer10_Features);
        }
        if (cpu_block_x > 0)
            ExecuteTenthLayer_CPU(Layer10_Weights_CPU, fire9_Features, Layer10_Features, gpu_block_x);
        if (gpu_block_x > 0)
            cudaDeviceSynchronize();
    } else {
        dim3 n_threads4(15, 1, 1);
        dim3 n_blocks4(15, 1, 1);
        ExecuteTenthLayer<<<n_blocks4, n_threads4>>>(Layer10_Weights_CPU, fire9_Features, Layer10_Features);
        cudaDeviceSynchronize();
    }
    t2 = gettime();

    cudaFree(Layer10_Weights_CPU);

    err = cudaGetLastError();

    // Global Avg pooling
    // ///////////////////////////////////////////////////////////////////////////////////

    double *output_GPU;
    err = cudaMalloc((void **)&output_GPU, 1000 * NUM * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate device data "
                "(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    double *output_CPU = (double *)malloc(1000 * NUM * sizeof(double));
    t1 = gettime();
    bool corun_20 = true;
    if (corun_20) {
        cpu_block_x = 1000 * cpu_offset / 100;
        gpu_block_x = 1000 - cpu_block_x;
        if (gpu_block_x > 0) {
            dim3 n_threads5(gpu_block_x, 1, 1);
            dim3 n_blocks5(1, 1, 1);
            global_pooling<<<n_blocks5, n_threads5>>>(Layer10_Features, output_GPU);
        }
        if (cpu_block_x > 0)
            global_pooling_CPU(Layer10_Features, output_CPU, gpu_block_x);
        if (gpu_block_x > 0) {
            cudaDeviceSynchronize();
            cudaMemcpy(output_CPU, output_GPU, gpu_block_x * sizeof(double), cudaMemcpyDeviceToHost);
        }
    } else {
        dim3 n_threads5(1000, 1, 1);
        dim3 n_blocks5(1, 1, 1);
        global_pooling<<<n_blocks5, n_threads5>>>(Layer10_Features, output_GPU);
        cudaDeviceSynchronize();
        cudaMemcpy(output_CPU, output_GPU, 1000 * sizeof(double), cudaMemcpyDeviceToHost);
    }
    t2 = gettime();
    double kernel_end_time = t2;

    cout << (kernel_end_time - kernel_start_time) << "\n";
    int predicted_class;
    predicted_class = predict_class(output_CPU);

    // Free functions
    cudaFree(Layer1_Features);
    cudaFree(Pool_Layer_Features);
    cudaFree(fire2squeeze1x1_Features);
    cudaFree(fire2_Features);
    cudaFree(fire3squeeze1x1_Features);
    cudaFree(fire3_Features);
    cudaFree(fire4squeeze1x1_Features);
    cudaFree(fire4_Features);
    cudaFree(Pool_Layer4_Features);
    cudaFree(fire5squeeze1x1_Features);
    cudaFree(fire5_Features);
    cudaFree(fire6squeeze1x1_Features);
    cudaFree(fire6_Features);
    cudaFree(fire7squeeze1x1_Features);
    cudaFree(fire7_Features);
    cudaFree(fire8squeeze1x1_Features_CPU);
    cudaFree(fire8_Features);
    cudaFree(Pool_Layer8_Features);
    cudaFree(fire9squeeze1x1_Features_CPU);
    cudaFree(fire9_Features);
    cudaFree(Layer10_Features);
    cudaFree(output_CPU);
}
