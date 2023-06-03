// +build ignore

#include "gs.h"
#include <json.hpp>
#include "common.h"
#include "llama.h"
#include <iostream>
#include <thread>
extern "C"
{
    extern void GsTokenCallback(uint32_t id, const char *buf);
    extern void GsFinishCallback(uint32_t id);
}

const size_t kvCacheSize = 512 * 1024 * 1024;
const int modelContextRollThreshold = 2000;
const int modelContextRollTarget = 1024;
const int taskPoolSize = 10;

gpt_params params;

llama_context *ctx = NULL;

Task currentTask = {0};

bool ctxDirty = true;
llama_token ctxTok[MODEL_MAX_TOKENS];
int ctxConsumedTokens = 0;

int gsInit(const char *modelPath)
{
    const char *device = "cpu";

    params.n_ctx = 2048;
    params.n_threads = 4;
    params.model = modelPath;
    params.n_batch = 128;

#ifdef GGML_USE_CUBLAS
    params.n_threads = 8;
    params.n_batch = 1024;
    params.n_gpu_layers = 1000;
    device = "cuda";
#endif

    ctx = llama_init_from_gpt_params(params);
    if (ctx == NULL)
    {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }
    printf("Model loaded\n");
    printf("Device: %s, modelPath: %s\n", device, modelPath);
    return 0;
}

int gsAddTask(uint32_t id, char *system, char *history, int maxNewTokens, int temperature, int topP)
{
    int ret = -1;
    auto systemPromptTokens = llama_tokenize(ctx, system, true);
    auto historyTokens = llama_tokenize(ctx, history, false);
    printf("id %d, system: %s, history: %s, maxNewTokens: %d, temperature: %d, topP: %d\n", id, system, history, maxNewTokens, temperature, topP);
    int historyLen = MODEL_MAX_TOKENS - 300 - systemPromptTokens.size();
    if (historyLen <= 0)
    {
        printf("id %d system too long, no enough space for history\n", id);
        goto final;
    }
    if (historyLen > historyTokens.size())
    {
        historyLen = historyTokens.size();
    }
    currentTask.taskID = id;
    currentTask.tokensGenerated = 0;
    currentTask.systemPromptLen = systemPromptTokens.size();
    currentTask.maxNewTokens = maxNewTokens;
    currentTask.temperature = temperature;
    currentTask.topP = topP;
    memcpy(currentTask.systemPromptTok, systemPromptTokens.data(), systemPromptTokens.size() * sizeof(llama_token));
    memcpy(currentTask.workTok, systemPromptTokens.data(), currentTask.systemPromptLen * sizeof(llama_token));
    memcpy(currentTask.workTok + currentTask.systemPromptLen, historyTokens.data() + historyTokens.size() - historyLen, historyLen * sizeof(llama_token));
    currentTask.workLen = currentTask.systemPromptLen + historyLen;
    currentTask.taskState = TASK_STATE_RUNNING;
    ctxDirty = true;
    ret = 0;
final:
    free(system);
    free(history);
    return ret;
}

int gsDoOnce()
{
    if (currentTask.taskState != TASK_STATE_RUNNING)
    {
        return 0;
    }

    if (currentTask.workLen >= modelContextRollThreshold)
    {
        printf("id %d roll context: %d\n", currentTask.taskID, currentTask.workLen);
        int nextHistoryLen = modelContextRollTarget - currentTask.systemPromptLen;
        if (nextHistoryLen > 0)
        {
            memmove(currentTask.workTok + currentTask.systemPromptLen, currentTask.workTok + currentTask.workLen - nextHistoryLen, nextHistoryLen * sizeof(llama_token));
        }
        currentTask.workLen = modelContextRollTarget;
        ctxDirty = true;
    }

    if (currentTask.tokensGenerated >= currentTask.maxNewTokens)
    {
        GsFinishCallback(currentTask.taskID);
        currentTask.taskState = TASK_STATE_INVALID;
        ctxDirty = true;
        return 0;
    }

    if (ctxDirty)
    {
        int tokensCanKeep = 0;
        for (int i = 0; i < ctxConsumedTokens; i++)
        {
            if (ctxTok[i] != currentTask.workTok[i])
            {
                break;
            }
            tokensCanKeep++;
        }
        printf("id %d tokensCanKeep %d, total: %d\nDifferences:\n", currentTask.taskID, tokensCanKeep, currentTask.workLen);
        for (int i = tokensCanKeep; i < currentTask.workLen; i++)
        {
            printf("%d %d %d\n", i, ctxTok[i], currentTask.workTok[i]);
        }
        int n_past = tokensCanKeep;
        while (n_past < currentTask.workLen)
        {
            int n_batch = std::min(params.n_batch, currentTask.workLen - n_past);
            printf("id %d n_past %d, n_batch %d\n", currentTask.taskID, n_past, n_batch);
            llama_eval(ctx, currentTask.workTok + n_past, n_batch, n_past, params.n_threads);
            n_past += n_batch;
        }
        memcpy(ctxTok, currentTask.workTok, currentTask.workLen * sizeof(llama_token));
        ctxConsumedTokens = currentTask.workLen;
        ctxDirty = false;
    }
    auto logits = llama_get_logits(ctx);
    auto n_vocab = llama_n_vocab(ctx);
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++)
    {
        candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
    }

    llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};
    auto nextToken = llama_sample_token_greedy(ctx, &candidates_p);
    printf("id %d nextToken %d\n", currentTask.taskID, nextToken);
    if (nextToken == llama_token_eos())
    {
        GsFinishCallback(currentTask.taskID);
        currentTask.taskState = TASK_STATE_INVALID;
        ctxDirty = true;
        llama_print_timings(ctx);
        return 0;
    }
    auto tokenStr = llama_token_to_str(ctx, nextToken);
    GsTokenCallback(currentTask.taskID, tokenStr);
    llama_eval(ctx, &nextToken, 1, currentTask.workLen, params.n_threads);
    currentTask.workTok[currentTask.workLen++] = nextToken;
    currentTask.tokensGenerated++;
    if (ctxConsumedTokens < MODEL_MAX_TOKENS) {
        ctxTok[ctxConsumedTokens++] = nextToken;
    } else {
        printf("!!! ctx full\n");
        ctxDirty = true;
    }
    return 1;
}

int gsCancelTask(uint32_t id)
{
    if (currentTask.taskID == id)
    {
        currentTask.taskState = TASK_STATE_INVALID;
        ctxDirty = true;
        return 0;
    }
    return -1;
}