<?php
// 图像采集页面的AJAX处理程序
session_start();

// 检查用户是否已登录
if (!isset($_SESSION['loggedin']) || $_SESSION['loggedin'] !== true) {
    header('Content-Type: application/json');
    echo json_encode(['error' => 'Not logged in']);
    exit;
}

// 检查是否是管理员
$is_admin = ($_SESSION['role'] ?? '') === 'admin';
// 获取当前登录用户的用户名
$current_username = $_SESSION['username'];

// 包含配置文件以获取数据库连接
include 'config.php';

// 如果不是管理员，只允许查询自己的目录状态
if (!$is_admin && isset($_POST['fullname'])) {
    // 尝试从数据库获取当前用户的全名
    try {
        // 使用config.php中已有的数据库连接

        // 查询当前用户的全名
        $stmt = $conn->prepare("SELECT fullname FROM users WHERE username = ?");
        $stmt->bind_param("s", $current_username);
        $stmt->execute();
        $result = $stmt->get_result();
        $user_data = $result->fetch_assoc();
        $stmt->close();

        if ($user_data && !empty($user_data['fullname'])) {
            $current_user_fullname = $user_data['fullname'];
        } else {
            // 如果在数据库中找不到，则使用会话中的全名或用户名
            $current_user_fullname = $_SESSION['fullname'] ?? $current_username;
        }
    } catch (Exception $e) {
        // 数据库查询失败，使用会话中的全名或用户名
        error_log("Database error: " . $e->getMessage());
        $current_user_fullname = $_SESSION['fullname'] ?? $current_username;
    }

    // 如果请求的不是当前用户的目录，拒绝访问
    if ($_POST['fullname'] !== $current_user_fullname) {
        header('Content-Type: application/json');
        echo json_encode(['error' => 'Permission denied']);
        exit;
    }
}

// 检查是否提供了fullname参数
if (!isset($_POST['fullname']) || empty($_POST['fullname'])) {
    header('Content-Type: application/json');
    echo json_encode(['error' => 'No fullname provided']);
    exit;
}

$fullname = trim($_POST['fullname']);

// 检查用户目录是否存在
$user_dir_exists = false;
$dataset_dir = '/FRAS/Image_DataSet';
$alt_dirs = ['../Image_DataSet', 'Image_DataSet', '/var/www/html/Image_DataSet'];
$user_dir = '';
$image_count = 0;
$next_image_number = 1;

// 记录当前工作目录和用户名，帮助调试
$current_dir = getcwd();
error_log("AJAX - Current working directory: " . $current_dir);
error_log("AJAX - Checking directories for user: " . $fullname);

// 检查主路径
if (file_exists($dataset_dir) && is_dir($dataset_dir)) {
    $user_dir = $dataset_dir . '/' . $fullname;
    error_log("AJAX - Checking primary path: " . $user_dir);
    if (file_exists($user_dir) && is_dir($user_dir)) {
        $user_dir_exists = true;
        error_log("AJAX - User directory found in primary path");
    }
}

// 检查备选路径
if (!$user_dir_exists) {
    foreach ($alt_dirs as $alt_dir) {
        $full_alt_dir = realpath($alt_dir);
        if ($full_alt_dir && is_dir($full_alt_dir)) {
            $user_dir = $full_alt_dir . '/' . $fullname;
            error_log("AJAX - Checking alternative path: " . $user_dir);
            if (file_exists($user_dir) && is_dir($user_dir)) {
                $user_dir_exists = true;
                error_log("AJAX - User directory found in alternative path: " . $full_alt_dir);
                break;
            }
        }
    }
}

// 如果仍未找到，尝试直接在当前目录下查找
if (!$user_dir_exists) {
    $local_dir = 'Image_DataSet/' . $fullname;
    error_log("AJAX - Checking local path: " . $local_dir);
    if (file_exists($local_dir) && is_dir($local_dir)) {
        $user_dir = $local_dir;
        $user_dir_exists = true;
        error_log("AJAX - User directory found in local path");
    }
}

// 如果目录存在，计算图片数量和下一个编号
if ($user_dir_exists && is_dir($user_dir)) {
    $files = scandir($user_dir);
    $image_files = [];
    $image_numbers = [];

    // 计算图片数量并提取编号
    foreach ($files as $file) {
        if (preg_match('/\.(jpg|jpeg|png)$/i', $file)) {
            $image_files[] = $file;

            // 提取编号
            if (preg_match('/^' . preg_quote($fullname, '/') . '_(\d+)_/', $file, $matches)) {
                if (isset($matches[1]) && is_numeric($matches[1])) {
                    $image_numbers[] = (int)$matches[1];
                }
            }
        }
    }

    $image_count = count($image_files);

    // 检查是否达到99张图片的限制
    $number_exceeded = ($image_count >= 99);

    // 确定下一个编号
    if (!empty($image_numbers)) {
        $next_image_number = max($image_numbers) + 1;

        // 如果编号超过99但总数未达到99，则重置为1
        if ($next_image_number > 99 && !$number_exceeded) {
            $next_image_number = 1;
        }
    }
}

// 返回结果
header('Content-Type: application/json');
echo json_encode([
    'exists' => $user_dir_exists,
    'image_count' => $image_count,
    'next_image_number' => $next_image_number,
    'number_exceeded' => $number_exceeded ?? false
]);
?>
