/**
 * @file rod_socket.c
 * @brief Socket communication module for ROD system
 * @author Noé Game
 * @date 15/02/2026
 * @see rod_socket.h
 * @copyright Cecill-C (Cf. LICENCE.txt)
 */

/* ******************************************************* Includes ****************************************************** */

#include "rod_socket.h"
#include "rod_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>
#include <fcntl.h>

/* ***************************************************** Public macros *************************************************** */

/* ************************************************** Public types definition ******************************************** */

/**
 * @brief Socket server context structure
 */
struct RodSocketServer {
    int socket_fd;      // Server socket file descriptor
    int client_fd;      // Connected client file descriptor (-1 if no client)
    char* socket_path;  // Path to Unix domain socket
};

/* *********************************************** Public functions declarations ***************************************** */

/* ******************************************* Public callback functions declarations ************************************ */

/* ********************************************* Function implementations *********************************************** */

RodSocketServer* rod_socket_server_create(const char* socket_path) {
    if (!socket_path) {
        fprintf(stderr, "rod_socket: socket_path is NULL\n");
        return NULL;
    }
    
    // Allocate server context
    RodSocketServer* server = (RodSocketServer*)malloc(sizeof(RodSocketServer));
    if (!server) {
        fprintf(stderr, "rod_socket: Failed to allocate server context\n");
        return NULL;
    }
    
    server->socket_fd = -1;
    server->client_fd = -1;
    server->socket_path = strdup(socket_path);
    if (!server->socket_path) {
        fprintf(stderr, "rod_socket: Failed to duplicate socket path\n");
        free(server);
        return NULL;
    }
    
    // Create Unix domain socket
    server->socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server->socket_fd < 0) {
        fprintf(stderr, "rod_socket: Failed to create socket: %s\n", strerror(errno));
        free(server->socket_path);
        free(server);
        return NULL;
    }
    
    // Remove existing socket file if it exists
    unlink(socket_path);
    
    // Setup socket address
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);
    
    // Bind socket
    if (bind(server->socket_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "rod_socket: Failed to bind socket to %s: %s\n", 
                socket_path, strerror(errno));
        close(server->socket_fd);
        free(server->socket_path);
        free(server);
        return NULL;
    }
    
    // Listen for connections (backlog = 1)
    if (listen(server->socket_fd, 1) < 0) {
        fprintf(stderr, "rod_socket: Failed to listen on socket: %s\n", strerror(errno));
        close(server->socket_fd);
        unlink(socket_path);
        free(server->socket_path);
        free(server);
        return NULL;
    }
    
    printf("rod_socket: Server listening on %s\n", socket_path);
    return server;
}

void rod_socket_server_destroy(RodSocketServer* server) {
    if (!server) return;
    
    // Close client connection
    if (server->client_fd >= 0) {
        close(server->client_fd);
        server->client_fd = -1;
    }
    
    // Close server socket
    if (server->socket_fd >= 0) {
        close(server->socket_fd);
        server->socket_fd = -1;
    }
    
    // Remove socket file
    if (server->socket_path) {
        unlink(server->socket_path);
        free(server->socket_path);
        server->socket_path = NULL;
    }
    
    free(server);
}

bool rod_socket_server_accept(RodSocketServer* server) {
    if (!server) return false;
    
    // If already have a client, nothing to do
    if (server->client_fd >= 0) {
        return true;
    }
    
    // Try to accept a connection (non-blocking)
    struct sockaddr_un client_addr;
    socklen_t client_len = sizeof(client_addr);
    
    // Set socket to non-blocking mode temporarily
    int flags = fcntl(server->socket_fd, F_GETFL, 0);
    fcntl(server->socket_fd, F_SETFL, flags | O_NONBLOCK);
    
    server->client_fd = accept(server->socket_fd, 
                               (struct sockaddr*)&client_addr, 
                               &client_len);
    
    // Restore blocking mode
    fcntl(server->socket_fd, F_SETFL, flags);
    
    if (server->client_fd >= 0) {
        printf("rod_socket: Client connected\n");
        return true;
    } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
        // No client waiting, not an error
        return true;
    } else {
        fprintf(stderr, "rod_socket: Error accepting client: %s\n", strerror(errno));
        return false;
    }
}

bool rod_socket_server_has_client(RodSocketServer* server) {
    return server && server->client_fd >= 0;
}

bool rod_socket_server_send_detections(RodSocketServer* server, 
                                        const MarkerData* markers, 
                                        int count) {
    if (!server) return false;
    
    // If no client connected, return success (no-op)
    if (server->client_fd < 0) {
        return true;
    }
    
    // Format detection results as JSON-like array: [[id, x, y, angle], ...]
    char buffer[ROD_MAX_DETECTION_SIZE];
    int offset = 0;
    
    // Start array
    offset += snprintf(buffer + offset, ROD_MAX_DETECTION_SIZE - offset, "[");
    
    // Add each marker
    for (int i = 0; i < count; i++) {
        if (i > 0) {
            offset += snprintf(buffer + offset, ROD_MAX_DETECTION_SIZE - offset, ",");
        }
        offset += snprintf(buffer + offset, ROD_MAX_DETECTION_SIZE - offset,
                          "[%d,%.2f,%.2f,%.4f]",
                          markers[i].id, markers[i].x, markers[i].y, markers[i].angle);
        
        // Check buffer overflow
        if (offset >= ROD_MAX_DETECTION_SIZE - 100) {
            fprintf(stderr, "rod_socket: Buffer too small, truncating results\n");
            break;
        }
    }
    
    // Close array and add newline
    offset += snprintf(buffer + offset, ROD_MAX_DETECTION_SIZE - offset, "]\n");
    
    // Print to console for debugging
    printf("rod_socket: Sending %d markers: %s", count, buffer);
    
    // Send via socket
    ssize_t total_sent = 0;
    ssize_t bytes_to_send = strlen(buffer);
    
    while (total_sent < bytes_to_send) {
        ssize_t sent = send(server->client_fd, buffer + total_sent, 
                          bytes_to_send - total_sent, MSG_NOSIGNAL);
        
        if (sent < 0) {
            if (errno == EPIPE || errno == ECONNRESET) {
                // Client disconnected
                printf("rod_socket: Client disconnected\n");
                close(server->client_fd);
                server->client_fd = -1;
                return false;
            } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Would block, try again
                usleep(1000);
                continue;
            } else {
                fprintf(stderr, "rod_socket: Error sending data: %s\n", strerror(errno));
                close(server->client_fd);
                server->client_fd = -1;
                return false;
            }
        }
        
        total_sent += sent;
    }
    
    return true;
}
