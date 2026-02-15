/**
 * @file rod_socket.h
 * @brief Socket communication module for ROD system
 * @author Noé Game
 * @date 15/02/2026
 * @see rod_socket.c
 * @copyright Cecill-C (Cf. LICENCE.txt)
 * 
 * This module provides Unix domain socket communication for sending
 * detection results from the CV thread to the communication thread.
 */

#pragma once

/* ******************************************************* Includes ****************************************************** */

#include "rod_cv.h"
#include <stdbool.h>
#include <stddef.h>

/* ***************************************************** Public macros *************************************************** */

/* ************************************************** Public types definition ******************************************** */

/**
 * @brief Opaque socket server context
 */
typedef struct RodSocketServer RodSocketServer;

/* *********************************************** Public functions declarations ***************************************** */

/**
 * @brief Create and initialize a socket server
 * @param socket_path Path to the Unix domain socket
 * @return Socket server context, or NULL on failure
 */
RodSocketServer* rod_socket_server_create(const char* socket_path);

/**
 * @brief Destroy socket server and cleanup resources
 * @param server Socket server context
 */
void rod_socket_server_destroy(RodSocketServer* server);

/**
 * @brief Try to accept a client connection (non-blocking)
 * @param server Socket server context
 * @return true if client connected or already connected, false on error
 */
bool rod_socket_server_accept(RodSocketServer* server);

/**
 * @brief Check if a client is currently connected
 * @param server Socket server context
 * @return true if client is connected, false otherwise
 */
bool rod_socket_server_has_client(RodSocketServer* server);

/**
 * @brief Send detection results to connected client
 * @param server Socket server context
 * @param markers Array of detected markers
 * @param count Number of markers
 * @return true on success, false on failure (client disconnected)
 * 
 * Formats detection results as JSON-like array: [[id, x, y, angle], ...]
 * If client is not connected, returns true (no-op).
 */
bool rod_socket_server_send_detections(RodSocketServer* server, 
                                        const MarkerData* markers, 
                                        int count);

#ifdef __cplusplus
}
#endif
