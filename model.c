#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "typedef.h"
#include "prng.h"
#include "prng.c"

typedef struct {
    u32 rows, cols;
    f32* data;
} matrix;

matrix* mat_create(u32 rows, u32 cols) {
    matrix* mat = (matrix*)calloc(1, sizeof(matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (f32*)calloc((u64)rows * cols, sizeof(f32));
    return mat;
}

void mat_free(matrix* mat) {
    if (mat == NULL) return;
    free(mat->data);
    free(mat);
}

matrix* mat_load(u32 rows, u32 cols, const char* filename) {
    matrix* mat = mat_create(rows, cols);

    FILE* f = fopen(filename, "rb");

    fseek(f, 0, SEEK_END);
    u64 size = ftell(f);
    fseek(f, 0, SEEK_SET);

    size = MIN(size, sizeof(f32) * rows * cols);

    fread(mat->data, 1, size, f);

    fclose(f);

    return mat;
}

b32 mat_copy(matrix* dst, matrix* src) {
    if (dst->rows != src->rows || dst->cols != src->cols) {
        return false;
    }

    memcpy(dst->data, src->data, sizeof(f32) * (u64)dst->rows * dst->cols);

    return true;
}

void mat_clear(matrix* mat) {
    memset(mat->data, 0, sizeof(f32) * (u64)mat->rows * mat->cols);
}

void mat_fill(matrix* mat, f32 x) {
    u64 size = (u64)mat->rows * mat->cols;
    for (u64 i = 0; i < size; i++) {
        mat->data[i] = x;
    }
}

void mat_fill_rand(matrix* mat, f32 lower, f32 upper) {
    u64 size = (u64)mat->rows * mat->cols;
    for (u64 i = 0; i < size; i++) {
        mat->data[i] = prng_randf() * (upper - lower) + lower;
    }
}

void mat_scale(matrix* mat, f32 scale) {
    u64 size = (u64)mat->rows * mat->cols;
    for (u64 i = 0; i < size; i++) {
        mat->data[i] *= scale;
    }
}

f32 mat_sum(matrix* mat) {
    u64 size = (u64)mat->rows * mat->cols;
    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++) {
        sum += mat->data[i];
    }
    return sum;
}

u64 mat_argmax(matrix* mat) {
    u64 size = (u64)mat->rows * mat->cols;
    u64 max_i = 0;
    for (u64 i = 0; i < size; i++) {
        if (mat->data[i] > mat->data[max_i]) {
            max_i = i;
        }
    }
    return max_i;
}

b32 mat_add(matrix* out, const matrix* a, const matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) { return false; }
    if (out->rows != a->rows || out->cols != a->cols) { return false; }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return true;
}

b32 mat_sub(matrix* out, const matrix* a, const matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) { return false; }
    if (out->rows != a->rows || out->cols != a->cols) { return false; }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
    return true;
}

void _mat_mul_nn(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 k = 0; k < a->cols; k++) {
            for (u64 j = 0; j < out->cols; j++) {
                out->data[j + i * out->cols] +=
                    a->data[k + i * a->cols] *
                    b->data[j + k * b->cols];
            }
        }
    }
}

void _mat_mul_nt(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 j = 0; j < out->cols; j++) {
            for (u64 k = 0; k < a->cols; k++) {
                out->data[j + i * out->cols] +=
                    a->data[k + i * a->cols] *
                    b->data[k + j * b->cols];
            }
        }
    }
}

void _mat_mul_tn(matrix* out, const matrix* a, const matrix* b) {
    for (u64 k = 0; k < a->rows; k++) {
        for (u64 i = 0; i < out->rows; i++) {
            for (u64 j = 0; j < out->cols; j++) {
                out->data[j + i * out->cols] +=
                    a->data[i + k * a->cols] *
                    b->data[j + k * b->cols];
            }
        }
    }
}

void _mat_mul_tt(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 j = 0; j < out->cols; j++) {
            for (u64 k = 0; k < a->rows; k++) {
                out->data[j + i * out->cols] +=
                    a->data[i + k * a->cols] *
                    b->data[k + j * b->cols];
            }
        }
    }
}

b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
) {
    u32 a_rows = transpose_a ? a->cols : a->rows;
    u32 a_cols = transpose_a ? a->rows : a->cols;
    u32 b_rows = transpose_b ? b->cols : b->rows;
    u32 b_cols = transpose_b ? b->rows : b->cols;

    if (a_cols != b_rows) { return false; }
    if (out->rows != a_rows || out->cols != b_cols) { return false; }

    if (zero_out) {
        mat_clear(out);
    }

    u32 transpose = (transpose_a << 1) | transpose_b;
    switch (transpose) {
        case 0b00: { _mat_mul_nn(out, a, b); } break;
        case 0b01: { _mat_mul_nt(out, a, b); } break;
        case 0b10: { _mat_mul_tn(out, a, b); } break;
        case 0b11: { _mat_mul_tt(out, a, b); } break;
    }

    return true;
}

b32 mat_relu(matrix* out, const matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) { return false; }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = MAX(0, in->data[i]);
    }
    return true;
}

b32 mat_softmax(matrix* out, const matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) { return false; }

    u64 size = (u64)out->rows * out->cols;

    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = expf(in->data[i]);
        sum += out->data[i];
    }

    mat_scale(out, 1.0f / sum);
    return true;
}

b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q) {
    if (p->rows != q->rows || p->cols != q->cols) { return false; }
    if (out->rows != p->rows || out->cols != p->cols) { return false; }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = p->data[i] == 0.0f ?
            0.0f : p->data[i] * -logf(q->data[i]);
    }
    return true;
}

b32 mat_relu_add_grad(matrix* out, const matrix* in, const matrix* grad) {
    if (out->rows != in->rows || out->cols != in->cols) { return false; }
    if (out->rows != grad->rows || out->cols != grad->cols) { return false; }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] += in->data[i] > 0.0f ? grad->data[i] : 0.0f;
    }
    return true;
}

b32 mat_softmax_add_grad(
    matrix* out, const matrix* softmax_out, const matrix* grad
) {
    if (softmax_out->rows != 1 && softmax_out->cols != 1) { return false; }

    u32 size = MAX(softmax_out->rows, softmax_out->cols);
    matrix* jacobian = mat_create(size, size);

    for (u32 i = 0; i < size; i++) {
        for (u32 j = 0; j < size; j++) {
            jacobian->data[j + i * size] =
                softmax_out->data[i] * ((i == j) - softmax_out->data[j]);
        }
    }

    mat_mul(out, jacobian, grad, 0, 0, 0);

    mat_free(jacobian);

    return true;
}

b32 mat_cross_entropy_add_grad(
    matrix* p_grad, matrix* q_grad,
    const matrix* p, const matrix* q, const matrix* grad
) {
    if (p->rows != q->rows || p->cols != q->cols) { return false; }

    u64 size = (u64)p->rows * p->cols;

    if (p_grad != NULL) {
        if (p_grad->rows != p->rows || p_grad->cols != p->cols) { return false; }
        for (u64 i = 0; i < size; i++) {
            p_grad->data[i] += -logf(q->data[i]) * grad->data[i];
        }
    }

    if (q_grad != NULL) {
        if (q_grad->rows != q->rows || q_grad->cols != q->cols) { return false; }
        for (u64 i = 0; i < size; i++) {
            q_grad->data[i] += -p->data[i] / q->data[i] * grad->data[i];
        }
    }

    return true;
}

typedef enum {
    MV_FLAG_HOME            = 0,
    MV_FLAG_REQUIRES_GRAD   = (1 << 0),
    MV_FLAG_PARAMETER       = (1 << 1),
    MV_FLAG_INPUT           = (1 << 2),
    MV_FLAG_OUTPUT          = (1 << 3),
    MV_FLAG_DESIRED_OUTPUT  = (1 << 4),
    MV_FLAG_COST            = (1 << 5),
} model_var_flags;

typedef enum {
    MV_OP_NULL = 0,
    MV_OP_CREATE,

    _MV_OP_UNARY_START,
    MV_OP_RELU,
    MV_OP_SOFTMAX,
    _MV_OP_BINARY_START,
    MV_OP_ADD,
    MV_OP_SUB,
    MV_OP_MATMUL,
    MV_OP_CROSS_ENTROPY,
} model_var_op;

#define MODEL_VAR_MAX_INPUTS 2
#define MV_NUM_INPUTS(op) ((op) < _MV_OP_UNARY_START ? 0 : ((op) < _MV_OP_BINARY_START ? 1 : 2))

typedef struct model_var {
    u32 index;
    u32 flags;

    matrix* val;
    matrix* grad;

    model_var_op op;
    struct model_var* inputs[MODEL_VAR_MAX_INPUTS];
} model_var;

typedef struct {
    model_var** vars;
    u32 size;
} model_program;

typedef struct {
    u32 num_vars;
    model_var* input;
    model_var* output;
    model_var* desired_output;
    model_var* cost;
    model_program forward_prog;
    model_program cost_prog;
} model_context;

typedef struct {
    matrix* train_images;
    matrix* train_labels;
    matrix* test_images;
    matrix* test_labels;
    u32 epochs;
    u32 batch_size;
    f32 learning_rate;
} model_training_desc;

model_var* mv_create(model_context* model, u32 rows, u32 cols, u32 flags) {
    model_var* out = (model_var*)calloc(1, sizeof(model_var));

    out->index = model->num_vars++;
    out->flags = flags;
    out->op = MV_OP_CREATE;
    out->val = mat_create(rows, cols);

    if (flags & MV_FLAG_REQUIRES_GRAD) {
        out->grad = mat_create(rows, cols);
    }

    if (flags & MV_FLAG_INPUT) { model->input = out; }
    if (flags & MV_FLAG_OUTPUT) { model->output = out; }
    if (flags & MV_FLAG_DESIRED_OUTPUT) { model->desired_output = out; }
    if (flags & MV_FLAG_COST) { model->cost = out; }

    return out;
}

model_var* _mv_unary_impl(
    model_context* model,
    model_var* input, u32 rows, u32 cols,
    u32 flags, model_var_op op
) {
    if (input->flags & MV_FLAG_REQUIRES_GRAD) {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    model_var* out = mv_create(model, rows, cols, flags);

    out->op = op;
    out->inputs[0] = input;

    return out;
}

model_var* _mv_binary_impl(
    model_context* model,
    model_var* a, model_var* b,
    u32 rows, u32 cols,
    u32 flags, model_var_op op
) {
    if (
        (a->flags & MV_FLAG_REQUIRES_GRAD) ||
        (b->flags & MV_FLAG_REQUIRES_GRAD)
    ) {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    model_var* out = mv_create(model, rows, cols, flags);

    out->op = op;
    out->inputs[0] = a;
    out->inputs[1] = b;

    return out;
}

model_var* mv_relu(model_context* model, model_var* input, u32 flags) {
    return _mv_unary_impl(
        model, input,
        input->val->rows, input->val->cols,
        flags, MV_OP_RELU
    );
}

model_var* mv_softmax(model_context* model, model_var* input, u32 flags) {
    return _mv_unary_impl(
        model, input,
        input->val->rows, input->val->cols,
        flags, MV_OP_SOFTMAX
    );
}

model_var* mv_add(model_context* model, model_var* a, model_var* b, u32 flags) {
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
        return NULL;
    }
    return _mv_binary_impl(
        model, a, b,
        a->val->rows, a->val->cols,
        flags, MV_OP_ADD
    );
}

model_var* mv_sub(model_context* model, model_var* a, model_var* b, u32 flags) {
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
        return NULL;
    }
    return _mv_binary_impl(
        model, a, b,
        a->val->rows, a->val->cols,
        flags, MV_OP_SUB
    );
}

model_var* mv_matmul(model_context* model, model_var* a, model_var* b, u32 flags) {
    if (a->val->cols != b->val->rows) {
        return NULL;
    }
    return _mv_binary_impl(
        model, a, b,
        a->val->rows, b->val->cols,
        flags, MV_OP_MATMUL
    );
}

model_var* mv_cross_entropy(model_context* model, model_var* p, model_var* q, u32 flags) {
    if (p->val->rows != q->val->rows || p->val->cols != q->val->cols) {
        return NULL;
    }
    return _mv_binary_impl(
        model, p, q,
        p->val->rows, p->val->cols,
        flags, MV_OP_CROSS_ENTROPY
    );
}

model_program model_prog_create(model_context* model, model_var* out_var) {
    b8* visited = (b8*)calloc(model->num_vars, sizeof(b8));
    model_var** stack = (model_var**)malloc(sizeof(model_var*) * model->num_vars);
    model_var** out = (model_var**)malloc(sizeof(model_var*) * model->num_vars);

    u32 stack_size = 0;
    u32 out_size = 0;

    stack[stack_size++] = out_var;

    while (stack_size > 0) {
        model_var* cur = stack[--stack_size];

        if (cur->index >= model->num_vars) { continue; }

        if (visited[cur->index]) {
            if (out_size < model->num_vars) {
                out[out_size++] = cur;
            }
            continue;
        }

        visited[cur->index] = true;

        if (stack_size < model->num_vars) {
            stack[stack_size++] = cur;
        }

        u32 num_inputs = MV_NUM_INPUTS(cur->op);
        for (u32 i = 0; i < num_inputs; i++) {
            model_var* input = cur->inputs[i];

            if (input->index >= model->num_vars || visited[input->index]) {
                continue;
            }

            for (u32 j = 0; j < stack_size; j++) {
                if (stack[j] == input) {
                    for (u32 k = j; k < stack_size-1; k++) {
                        stack[k] = stack[k+1];
                    }
                    stack_size--;
                }
            }

            if (stack_size < model->num_vars) {
                stack[stack_size++] = input;
            }
        }
    }

    model_program prog = {
        .size = out_size,
        .vars = (model_var**)malloc(sizeof(model_var*) * out_size)
    };

    memcpy(prog.vars, out, sizeof(model_var*) * out_size);

    free(visited);
    free(stack);
    free(out);

    return prog;
}

void model_prog_compute(model_program* prog) {
    for (u32 i = 0; i < prog->size; i++) {
        model_var* cur = prog->vars[i];

        model_var* a = cur->inputs[0];
        model_var* b = cur->inputs[1];

        switch (cur->op) {
            case MV_OP_NULL:
            case MV_OP_CREATE: break;

            case _MV_OP_UNARY_START: break;

            case MV_OP_RELU: { mat_relu(cur->val, a->val); } break;
            case MV_OP_SOFTMAX: { mat_softmax(cur->val, a->val); } break;

            case _MV_OP_BINARY_START: break;

            case MV_OP_ADD: { mat_add(cur->val, a->val, b->val); } break;
            case MV_OP_SUB: { mat_sub(cur->val, a->val, b->val); } break;
            case MV_OP_MATMUL: {
                mat_mul(cur->val, a->val, b->val, 1, 0, 0);
            } break;
            case MV_OP_CROSS_ENTROPY: {
                mat_cross_entropy(cur->val, a->val, b->val);
            } break;
        }
    }
}

void model_prog_compute_grads(model_program* prog) {
    for (u32 i = 0; i < prog->size; i++) {
        model_var* cur = prog->vars[i];

        if ((cur->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD) {
            continue;
        }

        if (cur->flags & MV_FLAG_PARAMETER) {
            continue;
        }

        mat_clear(cur->grad);
    }

    mat_fill(prog->vars[prog->size-1]->grad, 1.0f);

    for (i64 i = (i64)prog->size - 1; i >= 0; i--) {
        model_var* cur = prog->vars[i];

        if ((cur->flags & MV_FLAG_REQUIRES_GRAD) == 0) {
            continue;
        }

        model_var* a = cur->inputs[0];
        model_var* b = cur->inputs[1];

        u32 num_inputs = MV_NUM_INPUTS(cur->op);

        if (
            num_inputs == 1 &&
            (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD
        ) {
            continue;
        }

        if (
            num_inputs == 2 &&
            (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD &&
            (b->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD
        ) {
            continue;
        }

        switch (cur->op) {
            case MV_OP_NULL:
            case MV_OP_CREATE: break;

            case _MV_OP_UNARY_START: break;

            case MV_OP_RELU: {
                mat_relu_add_grad(a->grad, a->val, cur->grad);
            } break;
            case MV_OP_SOFTMAX: {
                mat_softmax_add_grad(a->grad, cur->val, cur->grad);
            } break;

            case _MV_OP_BINARY_START: break;

            case MV_OP_ADD: {
                if (a->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_add(a->grad, a->grad, cur->grad);
                }
                if (b->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_add(b->grad, b->grad, cur->grad);
                }
            } break;

            case MV_OP_SUB: {
                if (a->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_add(a->grad, a->grad, cur->grad);
                }
                if (b->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_sub(b->grad, b->grad, cur->grad);
                }
            } break;

            case MV_OP_MATMUL: {
                if (a->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_mul(a->grad, cur->grad, b->val, 0, 0, 1);
                }
                if (b->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_mul(b->grad, a->val, cur->grad, 0, 1, 0);
                }
            } break;

            case MV_OP_CROSS_ENTROPY: {
                model_var* p = a;
                model_var* q = b;
                mat_cross_entropy_add_grad(
                    p->grad, q->grad, p->val, q->val, cur->grad
                );
            } break;
        }
    }
}

model_context* model_create(void) {
    model_context* model = (model_context*)calloc(1, sizeof(model_context));
    return model;
}

void model_compile(model_context* model) {
    if (model->output != NULL) {
        model->forward_prog = model_prog_create(model, model->output);
    }

    if (model->cost != NULL) {
        model->cost_prog = model_prog_create(model, model->cost);
    }
}

void model_feedforward(model_context* model) {
    model_prog_compute(&model->forward_prog);
}

void model_destroy(model_context* model) {
    if (model == NULL) return;

    model_program* prog = &model->cost_prog;
    if (prog->vars != NULL) {
        for (u32 i = 0; i < prog->size; i++) {
            model_var* var = prog->vars[i];
            if (var->val) { mat_free(var->val); }
            if (var->grad) { mat_free(var->grad); }
            free(var);
        }
        free(prog->vars);
    }

    if (model->forward_prog.vars != NULL) {
        free(model->forward_prog.vars);
    }

    free(model);
}

void model_train(
    model_context* model,
    const model_training_desc* training_desc
) {
    matrix* train_images = training_desc->train_images;
    matrix* train_labels = training_desc->train_labels;
    matrix* test_images = training_desc->test_images;
    matrix* test_labels = training_desc->test_labels;

    u32 num_examples = train_images->rows;
    u32 input_size = train_images->cols;
    u32 output_size = train_labels->cols;
    u32 num_tests = test_images->rows;

    u32 num_batches = num_examples / training_desc->batch_size;

    u32* training_order = (u32*)malloc(sizeof(u32) * num_examples);
    for (u32 i = 0; i < num_examples; i++) {
        training_order[i] = i;
    }

    for (u32 epoch = 0; epoch < training_desc->epochs; epoch++) {
        for (u32 i = 0; i < num_examples; i++) {
            u32 a = prng_rand() % num_examples;
            u32 b = prng_rand() % num_examples;

            u32 tmp = training_order[b];
            training_order[b] = training_order[a];
            training_order[a] = tmp;
        }

        for (u32 batch = 0; batch < num_batches; batch++) {
            for (u32 i = 0; i < model->cost_prog.size; i++) {
                model_var* cur = model->cost_prog.vars[i];

                if (cur->flags & MV_FLAG_PARAMETER) {
                    mat_clear(cur->grad);
                }
            }

            f32 avg_cost = 0.0f;
            for (u32 i = 0; i < training_desc->batch_size; i++) {
                u32 order_index = batch * training_desc->batch_size + i;
                u32 index = training_order[order_index];

                memcpy(
                    model->input->val->data,
                    train_images->data + index * input_size,
                    sizeof(f32) * input_size
                );

                memcpy(
                    model->desired_output->val->data,
                    train_labels->data + index * output_size,
                    sizeof(f32) * output_size
                );

                model_prog_compute(&model->cost_prog);
                model_prog_compute_grads(&model->cost_prog);

                avg_cost += mat_sum(model->cost->val);
            }
            avg_cost /= (f32)training_desc->batch_size;

            for (u32 i = 0; i < model->cost_prog.size; i++) {
                model_var* cur = model->cost_prog.vars[i];

                if ((cur->flags & MV_FLAG_PARAMETER) != MV_FLAG_PARAMETER) {
                    continue;
                }

                mat_scale(
                    cur->grad,
                    training_desc->learning_rate /
                    training_desc->batch_size
                );
                mat_sub(cur->val, cur->val, cur->grad);
            }

            printf(
                "Epoch %2d / %2d, Batch %4d / %4d, Average Cost: %.4f\r",
                epoch + 1, training_desc->epochs,
                batch + 1, num_batches, avg_cost
            );
            fflush(stdout);
        }
        printf("\n");

        u32 num_correct = 0;
        f32 avg_cost = 0;
        for (u32 i = 0; i < num_tests; i++) {
            memcpy(
                model->input->val->data,
                test_images->data + i * input_size,
                sizeof(f32) * input_size
            );

            memcpy(
                model->desired_output->val->data,
                test_labels->data + i * output_size,
                sizeof(f32) * output_size
            );

            model_prog_compute(&model->cost_prog);

            avg_cost += mat_sum(model->cost->val);
            num_correct +=
                mat_argmax(model->output->val) ==
                mat_argmax(model->desired_output->val);
        }

        avg_cost /= (f32)num_tests;
        printf(
            "Test Completed. Accuracy: %5d / %5d (%.1f%%), Average Cost: %.4f\n",
            num_correct, num_tests, (f32)num_correct / num_tests * 100.0f,
            avg_cost
        );
    }

    free(training_order);
}
