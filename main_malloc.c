#define _CRT_SECURE_NO_WARNINGS

#include "model.c"

void draw_mnist_digit(f32* data);
void create_mnist_model(model_context* model);

int main(void) {
    matrix* train_images = mat_load(60000, 784, "train_images.mat");
    matrix* train_labels = mat_create(60000, 10);
    matrix* test_images = mat_load(10000, 784, "test_images.mat");
    matrix* test_labels = mat_create(10000, 10);

    {
        matrix* train_labels_file = mat_load(60000, 1, "train_labels.mat");
        matrix* test_labels_file = mat_load(10000, 1, "test_labels.mat");

        for (u32 i = 0; i < 60000; i++) {
            u32 num = train_labels_file->data[i];
            train_labels->data[i * 10 + num] = 1.0f;
        }

        for (u32 i = 0; i < 10000; i++) {
            u32 num = test_labels_file->data[i];
            test_labels->data[i * 10 + num] = 1.0f;
        }

        mat_free(train_labels_file);
        mat_free(test_labels_file);
    }

    draw_mnist_digit(test_images->data);
    for (u32 i = 0; i < 10; i++) {
        printf("%.0f ", test_labels->data[i]);
    }
    printf("\n\n");

    model_context* model = model_create();
    create_mnist_model(model);
    model_compile(model);

    memcpy(model->input->val->data, test_images->data, sizeof(f32) * 784);
    model_feedforward(model);

    printf("pre-training output: ");
    for (u32 i = 0; i < 10; i++) {
        printf("%.2f", model->output->val->data[i]);
    }
    printf("\n");

    model_training_desc training_desc = {
        .train_images = train_images,
        .train_labels = train_labels,
        .test_images = test_images,
        .test_labels = test_labels,

        .epochs = 10,
        .batch_size = 50,
        .learning_rate = 0.01f
    };
    model_train(model, &training_desc);

    memcpy(model->input->val->data, test_images->data, sizeof(f32) * 784);
    model_feedforward(model);
    printf("post-training output: ");
    for (u32 i = 0; i < 10; i++) {
        printf("%f ", model->output->val->data[i]);
    }
    printf("\n\n");

    model_destroy(model);
    mat_free(train_images);
    mat_free(train_labels);
    mat_free(test_images);
    mat_free(test_labels);

    return 0;
}

void draw_mnist_digit(f32* data) {
    for (u32 y = 0; y < 28; y++) {
        for (u32 x = 0; x < 28; x++) {
            f32 num = data[x + y * 28];
            u32 col = 232 + (u32)(num * 23);
            printf("\x1b[48;5;%dm  ", col);
        }
        printf("\n");
    }
    printf("\x1b[0m");
}

void create_mnist_model(model_context* model) {
    model_var* input = mv_create(model, 784, 1, MV_FLAG_INPUT);

    model_var* W0 = mv_create(model, 16, 784, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* W1 = mv_create(model, 16, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* W2 = mv_create(model, 10, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    f32 bound0 = sqrtf(6.0f / (784 + 16));
    f32 bound1 = sqrtf(6.0f / (16 + 16));
    f32 bound2 = sqrtf(6.0f / (16 + 10));
    mat_fill_rand(W0->val, -bound0, bound0);
    mat_fill_rand(W1->val, -bound1, bound1);
    mat_fill_rand(W2->val, -bound2, bound2);

    model_var* b0 = mv_create(model, 16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* b1 = mv_create(model, 16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* b2 = mv_create(model, 10, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    model_var* z0_a = mv_matmul(model, W0, input, 0);
    model_var* z0_b = mv_add(model, z0_a, b0, 0);
    model_var* a0 = mv_relu(model, z0_b, 0);

    model_var* z1_a = mv_matmul(model, W1, a0, 0);
    model_var* z1_b = mv_add(model, z1_a, b1, 0);
    model_var* z1_c = mv_relu(model, z1_b, 0);
    model_var* a1 = mv_add(model, a0, z1_c, 0);

    model_var* z2_a = mv_matmul(model, W2, a1, 0);
    model_var* z2_b = mv_add(model, z2_a, b2, 0);
    model_var* output = mv_softmax(model, z2_b, MV_FLAG_OUTPUT);

    model_var* y = mv_create(model, 10, 1, MV_FLAG_DESIRED_OUTPUT);

    model_var* cost = mv_cross_entropy(model, y, output, MV_FLAG_COST);
}
