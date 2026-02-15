// server.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <stdint.h>

#define SOCKET_PATH "./tmp/python_c_socket"

int main() {
    int server_fd, client_fd;
    struct sockaddr_un addr;

    unlink(SOCKET_PATH);

    server_fd = socket(AF_UNIX, SOCK_STREAM, 0);

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, SOCKET_PATH);

    bind(server_fd, (struct sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 1);

    printf("En attente de connexion...\n");
    client_fd = accept(server_fd, NULL, NULL);

    int32_t width, height, n_int, n_float;

    read(client_fd, &width, sizeof(int32_t));
    read(client_fd, &height, sizeof(int32_t));
    read(client_fd, &n_int, sizeof(int32_t));
    read(client_fd, &n_float, sizeof(int32_t));

    printf("Image: %dx%d\n", width, height);
    printf("Nb int: %d\n", n_int);
    printf("Nb float: %d\n", n_float);

    uint8_t* image = malloc(width * height);
    read(client_fd, image, width * height);

    int32_t* int_list = malloc(n_int * sizeof(int32_t));
    read(client_fd, int_list, n_int * sizeof(int32_t));

    float* float_list = malloc(n_float * sizeof(float));
    read(client_fd, float_list, n_float * sizeof(float));

    printf("Premier pixel: %d\n", image[0]);
    printf("Premier int: %d\n", int_list[0]);
    printf("Premier float: %f\n", float_list[0]);

    free(image);
    free(int_list);
    free(float_list);

    close(client_fd);
    close(server_fd);
    unlink(SOCKET_PATH);

    return 0;
}
