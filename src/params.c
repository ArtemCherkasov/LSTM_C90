#include "params.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void parse_parameters_file(const char *path, ParamsFile *pf) {
    FILE *f;
    char line[512];
    char *eq_pos;
    char key[128];
    char val[256];

    memset(pf, 0, sizeof(ParamsFile));

    f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Warning: cannot open parameters file %s, using defaults.\n", path);
        return;
    }

    while (fgets(line, sizeof(line), f)) {
        utils_trim(line);
        if (line[0] == '#' || line[0] == 0) continue;

        eq_pos = strchr(line, '=');
        if (!eq_pos) continue;

        {
            int klen = (int)(eq_pos - line);
            if (klen >= 128) klen = 127;
            memcpy(key, line, (size_t)klen);
            key[klen] = 0;
            utils_trim(key);
        }
        {
            strcpy(val, eq_pos + 1);
            utils_trim(val);
        }

        if (val[0] == 0) continue; /* blank value = use default */

        if (strcmp(key, "cells") == 0) { pf->cells = atoi(val); pf->has_cells = 1; }
        else if (strcmp(key, "epochs") == 0) { pf->epochs = atoi(val); pf->has_epochs = 1; }
        else if (strcmp(key, "lr") == 0) { pf->lr = atof(val); pf->has_lr = 1; }
        else if (strcmp(key, "seq") == 0) { pf->seq = atoi(val); pf->has_seq = 1; }
        else if (strcmp(key, "mode") == 0) { strncpy(pf->mode, val, 31); pf->mode[31]=0; pf->has_mode = 1; }
        else if (strcmp(key, "output") == 0) { pf->output = atoi(val); pf->has_output = 1; }
        else if (strcmp(key, "datetime") == 0) { strncpy(pf->datetime, val, 31); pf->datetime[31]=0; pf->has_datetime = 1; }
        else if (strcmp(key, "from") == 0) { strncpy(pf->from, val, 31); pf->from[31]=0; pf->has_from = 1; }
        else if (strcmp(key, "to") == 0) { strncpy(pf->to, val, 31); pf->to[31]=0; pf->has_to = 1; }
        else if (strcmp(key, "data") == 0) { strncpy(pf->data, val, 255); pf->data[255]=0; pf->has_data = 1; }
        else if (strcmp(key, "model") == 0) { strncpy(pf->model, val, 255); pf->model[255]=0; pf->has_model = 1; }
        else if (strcmp(key, "h") == 0) { pf->h = atoi(val); pf->has_h = 1; }
        else if (strcmp(key, "wd") == 0) { pf->wd = atof(val); pf->has_wd = 1; }
        else if (strcmp(key, "patience") == 0) { pf->patience = atoi(val); pf->has_patience = 1; }
        else if (strcmp(key, "sdata") == 0) { strncpy(pf->sdata, val, 255); pf->sdata[255]=0; pf->has_sdata = 1; }
        else if (strcmp(key, "batch_size") == 0) { pf->batch_size = atoi(val); pf->has_batch_size = 1; }
        else if (strcmp(key, "momentum") == 0) { pf->momentum = atof(val); pf->has_momentum = 1; }
        else if (strcmp(key, "seed") == 0) { pf->seed = (unsigned int)atoi(val); pf->has_seed = 1; }
        else if (strcmp(key, "clip_min") == 0) { pf->clip_min = atof(val); pf->has_clip_min = 1; }
        else if (strcmp(key, "clip_max") == 0) { pf->clip_max = atof(val); pf->has_clip_max = 1; }
        else if (strcmp(key, "train_ratio") == 0) { pf->train_ratio = atof(val); pf->has_train_ratio = 1; }
        else if (strcmp(key, "volume_return_clip") == 0) { pf->volume_return_clip = atof(val); pf->has_volume_return_clip = 1; }
        else if (strcmp(key, "forget_gate_bias_init") == 0) { pf->forget_gate_bias_init = atof(val); pf->has_forget_gate_bias_init = 1; }
        else if (strcmp(key, "prediction_threshold") == 0) { pf->prediction_threshold = atof(val); pf->has_prediction_threshold = 1; }
    }
    fclose(f);
}
