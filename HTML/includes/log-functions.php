<?php
/**
 * 系统日志函数 - 用于记录系统日志
 */

/**
 * 添加系统日志
 *
 * @param string $message 日志消息
 * @param string $level 日志级别 ('info', 'warning', 'error', 'debug')
 * @param string $source 日志来源
 * @param int $user_id 用户ID，如果有的话
 * @return bool 是否成功添加日志
 */
function add_system_log($message, $level = 'info', $source = null, $user_id = null) {
    global $conn;

    // 假设系统日志表已经存在
    // 如果表不存在，日志将无法记录，但不会影响系统运行

    // 如果未提供用户ID，但用户已登录，则使用当前用户ID
    if ($user_id === null && isset($_SESSION['id'])) {
        $user_id = $_SESSION['id'];
    }

    // 准备SQL语句
    $sql = "INSERT INTO system_logs (log_level, message, source, user_id) VALUES (?, ?, ?, ?)";
    $stmt = $conn->prepare($sql);

    if (!$stmt) {
        error_log("Error preparing log statement: " . $conn->error);
        return false;
    }

    $stmt->bind_param("sssi", $level, $message, $source, $user_id);
    $result = $stmt->execute();
    $stmt->close();

    return $result;
}

/**
 * 添加信息级别日志
 *
 * @param string $message 日志消息
 * @param string $source 日志来源
 * @param int $user_id 用户ID，如果有的话
 * @return bool 是否成功添加日志
 */
function log_info($message, $source = null, $user_id = null) {
    return add_system_log($message, 'info', $source, $user_id);
}

/**
 * 添加警告级别日志
 *
 * @param string $message 日志消息
 * @param string $source 日志来源
 * @param int $user_id 用户ID，如果有的话
 * @return bool 是否成功添加日志
 */
function log_warning($message, $source = null, $user_id = null) {
    return add_system_log($message, 'warning', $source, $user_id);
}

/**
 * 添加错误级别日志
 *
 * @param string $message 日志消息
 * @param string $source 日志来源
 * @param int $user_id 用户ID，如果有的话
 * @return bool 是否成功添加日志
 */
function log_error($message, $source = null, $user_id = null) {
    return add_system_log($message, 'error', $source, $user_id);
}

/**
 * 添加调试级别日志
 *
 * @param string $message 日志消息
 * @param string $source 日志来源
 * @param int $user_id 用户ID，如果有的话
 * @return bool 是否成功添加日志
 */
function log_debug($message, $source = null, $user_id = null) {
    return add_system_log($message, 'debug', $source, $user_id);
}
?>
