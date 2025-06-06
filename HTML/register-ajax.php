<?php
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

// 设置缓存控制头，防止浏览器缓存
header("Cache-Control: no-store, no-cache, must-revalidate, max-age=0");
header("Cache-Control: post-check=0, pre-check=0", false);
header("Pragma: no-cache");

// 设置响应头为 JSON
header('Content-Type: application/json');

session_start();
include 'config.php';
include 'includes/log-functions.php';
include 'includes/language-loader.php';

$response = ['success' => false, 'message' => ''];

// 如果用户已登录，返回错误
if (isset($_SESSION['loggedin']) && $_SESSION['loggedin'] === true) {
    $response['message'] = __('already_logged_in');
    echo json_encode($response);
    exit;
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // 获取表单数据
    $username = $_POST['username'];
    $fullname = $_POST['fullname'];
    $password = $_POST['password'];
    $confirm_password = $_POST['confirm_password'];

    // 验证数据
    if (empty($username) || empty($fullname) || empty($password) || empty($confirm_password)) {
        $response['message'] = __('all_fields_required');
    } elseif ($password !== $confirm_password) {
        $response['message'] = __('password_mismatch');
    } else {
        // 检查用户名是否已存在
        $check_username_sql = "SELECT username FROM users WHERE username = ?";
        $check_username_stmt = $conn->prepare($check_username_sql);
        $check_username_stmt->bind_param("s", $username);
        $check_username_stmt->execute();
        $check_username_result = $check_username_stmt->get_result();

        if ($check_username_result->num_rows > 0) {
            $response['message'] = __('username_exists');
        } else {
            // 开始事务
            $conn->begin_transaction();

            try {
                // 检查是否有裁剪后的图片数据
                if (empty($_POST['cropped_image'])) {
                    throw new Exception(__('profile_picture_required'));
                }

                // 从base64字符串中提取图片数据
                $cropped_image = $_POST['cropped_image'];
                $image_parts = explode(";base64,", $cropped_image);

                // 确保数据格式正确
                if (count($image_parts) !== 2) {
                    throw new Exception(__('invalid_image_format'));
                }

                $image_data_base64 = $image_parts[1];
                $image_data = base64_decode($image_data_base64);

                // 获取员工ID
                $employee_id = $_POST['employee_id'];

                // 检查员工ID是否已存在
                $check_employee_id_sql = "SELECT employee_id FROM users WHERE employee_id = ?";
                $check_employee_id_stmt = $conn->prepare($check_employee_id_sql);
                $check_employee_id_stmt->bind_param("i", $employee_id);
                $check_employee_id_stmt->execute();
                $check_employee_id_result = $check_employee_id_stmt->get_result();

                if ($check_employee_id_result->num_rows > 0) {
                    throw new Exception(__('employee_id_exists'));
                }

                // 设置默认角色为用户
                $role = "user";

                // 对密码进行哈希处理
                $hashed_password = password_hash($password, PASSWORD_DEFAULT);

                // 插入新用户
                $insert_sql = "INSERT INTO users (employee_id, username, password, fullname, role, profile_picture) VALUES (?, ?, ?, ?, ?, ?)";
                $insert_stmt = $conn->prepare($insert_sql);
                $insert_stmt->bind_param("isssss", $employee_id, $username, $hashed_password, $fullname, $role, $image_data);
                $insert_stmt->execute();

                // 创建文件名 - 使用用户的全名
                $filename = $fullname . '.jpg';
                // 只替换文件名中的特殊字符，保留空格
                $filename = preg_replace('/[^a-zA-Z0-9_\-\. ]/', '', $filename);
                $profile_pictures_dir = '/home/xiaxialaolao/FRAS_env/Profile_Pictures';
                $filepath = $profile_pictures_dir . '/' . $filename;

                // 确保目录存在
                if (!file_exists($profile_pictures_dir)) {
                    mkdir($profile_pictures_dir, 0755, true);
                }

                // 检查并删除旧的图片文件（如果存在）
                if (file_exists($filepath)) {
                    unlink($filepath);
                }

                // 保存图片到文件系统
                $result = file_put_contents($filepath, $image_data);

                if ($result !== false) {
                    // 确保文件权限正确
                    chmod($filepath, 0644);

                    // 强制文件系统同步，确保更改立即可见
                    clearstatcache(true, $filepath);
                }

                // 提交事务
                $conn->commit();

                // 记录用户注册日志
                $log_message = "New user registered: $username";
                log_info($log_message, "register-ajax.php");

                $response['success'] = true;
                $response['message'] = __('registration_successful');

            } catch (Exception $e) {
                // 回滚事务
                $conn->rollback();
                $response['message'] = $e->getMessage();
            }
        }
    }
}

echo json_encode($response);
