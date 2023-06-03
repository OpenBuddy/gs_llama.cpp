#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif 

#define MODEL_MAX_TOKENS (2048)

#define TASK_STATE_INVALID (0)
#define TASK_STATE_RUNNING (1)

struct Task {
    int taskState;
    uint32_t taskID;

    int maxNewTokens;
    int tokensGenerated;
    int temperature; // 100
    int topP; // 100

    int workTok[MODEL_MAX_TOKENS];
    int workLen;
    int systemPromptTok[MODEL_MAX_TOKENS];
    int systemPromptLen;

};

// this is an so file, export functions: gsInit
int gsInit(const char* modelPath);


int gsAddTask(uint32_t id, char* system, char* history, int maxNewTokens, int temperature, int topP);

int gsCancelTask(uint32_t id) ;

int gsDoOnce();




#ifdef __cplusplus
}
#endif