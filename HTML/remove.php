<?php
session_start();
include 'config.php';
include 'includes/language-loader.php';
include 'includes/notification-functions.php';
include 'includes/log-functions.php';

// 未登录重定向
if (!isset($_SESSION['loggedin']) || $_SESSION['loggedin'] !== true) {
    header("Location: login.php");
    exit();
}

// 检查是否为管理员
if ($_SESSION['role'] !== 'admin') {
    header("Location: dashboard.php");
    exit();
}

$error = "";
$success = "";

// 检查是否有ID参数
if (!isset($_GET['id']) || empty($_GET['id'])) {
    header("Location: dashboard.php");
    exit();
}

$employee_id = $_GET['id'];

// 确认删除操作
if (isset($_POST['confirm_delete']) && $_POST['confirm_delete'] === 'yes') {
    // 开始事务
    $conn->begin_transaction();

    try {
        // 首先获取用户完整信息（用于日志记录和删除头像文件）
        $get_user_sql = "SELECT employee_id, username, fullname, role FROM users WHERE employee_id = ?";
        $get_user_stmt = $conn->prepare($get_user_sql);
        $get_user_stmt->bind_param("i", $employee_id);
        $get_user_stmt->execute();
        $user_result = $get_user_stmt->get_result();

        if ($user_result->num_rows === 0) {
            throw new Exception("User not found");
        }

        $user = $user_result->fetch_assoc();
        $fullname = $user['fullname'];

        // 删除用户头像文件
        $filename = preg_replace('/[^a-zA-Z0-9_\-\. ]/', '', $fullname) . '.jpg';
        $profile_pictures_dir = '/home/xiaxialaolao/FRAS_env/Profile_Pictures';
        $filepath = $profile_pictures_dir . '/' . $filename;

        // 如果文件存在，则删除
        if (file_exists($filepath)) {
            unlink($filepath);
        }

        // 首先删除用户设置表中的记录
        $delete_settings_sql = "DELETE FROM users_settings WHERE employee_id = ?";
        $delete_settings_stmt = $conn->prepare($delete_settings_sql);
        $delete_settings_stmt->bind_param("i", $employee_id);
        $delete_settings_stmt->execute();

        // 删除考勤记录 - 使用正确的表名 attendance
        $delete_attendance_sql = "DELETE FROM attendance WHERE employee_id = ?";
        $delete_attendance_stmt = $conn->prepare($delete_attendance_sql);
        $delete_attendance_stmt->bind_param("i", $employee_id);
        $delete_attendance_stmt->execute();

        // 更新系统日志中的用户ID引用为NULL（因为有ON DELETE SET NULL）
        $update_logs_sql = "UPDATE system_logs SET user_id = NULL WHERE user_id = ?";
        $update_logs_stmt = $conn->prepare($update_logs_sql);
        $update_logs_stmt->bind_param("i", $employee_id);
        $update_logs_stmt->execute();

        // 最后删除用户记录
        $delete_sql = "DELETE FROM users WHERE employee_id = ?";
        $delete_stmt = $conn->prepare($delete_sql);
        $delete_stmt->bind_param("i", $employee_id);
        $delete_stmt->execute();

        if ($delete_stmt->affected_rows === 0) {
            throw new Exception("Failed to delete user");
        }

        // 记录删除的相关表信息
        log_info("Deleted related records for user ID: {$employee_id}", "remove.php");

        // 记录用户删除日志
        $log_message = "User deleted: {$user['username']}";
        log_warning($log_message, "remove.php");

        // 提交事务
        $conn->commit();

        // 设置成功通知并重定向
        $success_message = __('user_deleted');
        set_success_notification($success_message);

        // 重定向回仪表板
        header("Location: dashboard.php");
        exit();

    } catch (Exception $e) {
        // 回滚事务
        $conn->rollback();

        $error = $e->getMessage();
        set_error_notification($error);
    }
} else {
    // 获取用户信息以显示确认页面
    $get_user_sql = "SELECT employee_id, username, fullname, role, profile_picture FROM users WHERE employee_id = ?";
    $get_user_stmt = $conn->prepare($get_user_sql);
    $get_user_stmt->bind_param("i", $employee_id);
    $get_user_stmt->execute();
    $user_result = $get_user_stmt->get_result();

    if ($user_result->num_rows === 0) {
        header("Location: dashboard.php");
        exit();
    }

    $user = $user_result->fetch_assoc();
}
?>
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="icon.png" type="image/x-icon">
    <link rel="stylesheet" href="css/dashboard-style.css">
    <link rel="stylesheet" href="css/slidebar.css">
    <link rel="stylesheet" href="css/remove-style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <?php
    include 'includes/theme-loader.php';
    include 'includes/date-formatter.php';
    include 'includes/notifications-loader.php';
    ?>
    <title><?php echo __('delete_user'); ?> - FRAS System</title>
</head>
<body>
<div class="dashboard-container">
        <?php include 'includes/sidebar.php'; ?>

        <!-- 主内容区 -->
        <div class="main-content">
            <!-- 顶部标题区 -->
            <div class="content-header">
                <h2 class="content-title"><?php echo __('delete_user'); ?></h2>

                <button onclick="window.location.href='dashboard.php'" class="action-button">
                    <i class="fas fa-arrow-left"></i>
                    <?php echo __('back_to_dashboard'); ?>
                </button>
            </div>

            <!-- 删除确认区域 -->
            <div class="delete-container">
                <!-- 通知由 notifications-loader.php 处理 -->

                <div class="warning-icon">
                    <i class="fas fa-exclamation-triangle"></i>
                </div>

                <h2 class="delete-title"><?php echo __('delete_user_confirmation'); ?></h2>

                <p class="delete-message">
                    <?php echo __('delete_user_warning'); ?><br>
                    <?php echo __('delete_user_warning_data'); ?>
                </p>

                <div class="user-info">
                    <div class="user-avatar">
                        <?php if (!empty($user['profile_picture'])): ?>
                            <img src="data:image/jpeg;base64,<?php echo base64_encode($user['profile_picture']); ?>" alt="Profile Picture">
                        <?php else: ?>
                            <img src="default_avatar.png" alt="Default Profile Picture">
                        <?php endif; ?>
                    </div>

                    <div class="user-info-item">
                        <div class="user-info-label"><?php echo __('employee_id'); ?>:</div>
                        <div class="user-info-value"><?php echo htmlspecialchars($user['employee_id']); ?></div>
                    </div>

                    <div class="user-info-item">
                        <div class="user-info-label"><?php echo __('username'); ?>:</div>
                        <div class="user-info-value"><?php echo htmlspecialchars($user['username']); ?></div>
                    </div>

                    <div class="user-info-item">
                        <div class="user-info-label"><?php echo __('full_name'); ?>:</div>
                        <div class="user-info-value"><?php echo htmlspecialchars($user['fullname']); ?></div>
                    </div>

                    <div class="user-info-item">
                        <div class="user-info-label"><?php echo __('role'); ?>:</div>
                        <div class="user-info-value"><?php echo $user['role'] == 'admin' ? __('admin') : __('user'); ?></div>
                    </div>
                </div>

                <form method="POST">
                    <input type="hidden" name="confirm_delete" value="yes">

                    <div class="button-group">
                        <a href="dashboard.php" class="cancel-button">
                            <i class="fas fa-times"></i> <?php echo __('cancel'); ?>
                        </a>
                        <button type="submit" class="delete-button">
                            <i class="fas fa-trash"></i> <?php echo __('delete_user'); ?>
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</body>
</html>
